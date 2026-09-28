"""
API views — thin HTTP layer over the inference services.

Every view keeps validation and model access at arm's length: serializers guard
single predictions, `validate_batch_frame` guards CSV uploads, and
`api_exception_handler` guarantees clients only ever see structured JSON — never a
Django stack trace, even when DEBUG is on.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from django.conf import settings

from .schema import _clean, get_field_metadata
from .serializers import PredictionInputSerializer, validate_batch_frame
from .services import predict_batch, predict_one

logger = logging.getLogger(__name__)


def api_exception_handler(exc, context):
    """Ensure clients receive clean JSON, never an HTML stack trace.

    Delegates to DRF for handled cases (validation, 404, throttling); any
    unhandled exception is logged server-side and returned as a generic 500.
    """
    response = drf_exception_handler(exc, context)
    if response is not None:
        return response
    logger.exception("Unhandled API exception", exc_info=exc)
    return Response(
        {"detail": "An internal error occurred. Please try again."},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@api_view(["POST"])
def predict(request):
    """Single-record prediction. Validates all 38 features, then runs the model."""
    serializer = PredictionInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({"errors": serializer.errors},
                        status=status.HTTP_400_BAD_REQUEST)
    result = predict_one(serializer.validated_data)
    return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
@parser_classes([MultiPartParser])
def predict_batch_view(request):
    """Batch prediction from an uploaded CSV. Size-, format- and schema-checked."""
    upload = request.FILES.get("file")
    if upload is None:
        return Response({"detail": "No file uploaded. Send a CSV as 'file'."},
                        status=status.HTTP_400_BAD_REQUEST)

    if upload.size > settings.MAX_UPLOAD_BYTES:
        limit_mb = settings.MAX_UPLOAD_BYTES / (1024 * 1024)
        return Response(
            {"detail": f"File too large. Maximum size is {limit_mb:.0f} MB."},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    if not upload.name.lower().endswith(".csv"):
        return Response({"detail": "Only .csv files are accepted."},
                        status=status.HTTP_400_BAD_REQUEST)

    try:
        raw = upload.read().decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(raw))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return Response({"detail": "Could not parse the file as CSV."},
                        status=status.HTTP_400_BAD_REQUEST)

    if df.empty:
        return Response({"detail": "The uploaded CSV has no rows."},
                        status=status.HTTP_400_BAD_REQUEST)

    df = _clean(df)
    missing, category_errors = validate_batch_frame(df)
    if missing:
        return Response(
            {"detail": "The CSV is missing required columns.",
             "missing_columns": missing},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if category_errors:
        return Response(
            {"detail": "The CSV contains unrecognized categorical values.",
             "category_errors": category_errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    results = predict_batch(df)
    return Response({"count": len(results), "results": results},
                    status=status.HTTP_200_OK)


@api_view(["GET"])
def metadata(request):
    """Field metadata powering the frontend's dropdowns, sliders and defaults."""
    return Response(get_field_metadata(), status=status.HTTP_200_OK)


@api_view(["GET"])
def metrics(request):
    """Holdout metrics for the deployed model plus every benchmarked model."""
    from .model_loader import get_metrics

    return Response(get_metrics(), status=status.HTTP_200_OK)
