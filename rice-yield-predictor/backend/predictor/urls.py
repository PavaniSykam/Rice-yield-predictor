"""URL routes for the predictor API (mounted under /api/ by config.urls)."""
from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("predict/", views.predict, name="predict"),
    path("predict/batch/", views.predict_batch_view, name="predict-batch"),
    path("metadata/", views.metadata, name="metadata"),
    path("metrics/", views.metrics, name="metrics"),
]
