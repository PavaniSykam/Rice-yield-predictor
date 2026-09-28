"""
DRF serializers — validate input BEFORE it can reach the ML model.

Field names are built dynamically because several (e.g. "30DRain__in_mm") are not
valid Python identifiers. Categorical fields use ChoiceField (values seen in training);
numeric fields are floats bounded by the dataset range with generous padding.
"""
from __future__ import annotations

import math

from rest_framework import serializers

from .schema import CAT_COLS, NUM_COLS, get_field_metadata

# Numeric inputs may exceed the observed range by this fraction before rejection.
# The frontend sliders pad by ~15%, so legitimate UI values always pass.
_RANGE_PAD = 0.5


class PredictionInputSerializer(serializers.Serializer):
    """All model features required; categoricals constrained to known values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = get_field_metadata()
        for col in CAT_COLS:
            choices = meta["categorical"].get(col, [])
            self.fields[col] = serializers.ChoiceField(
                choices=choices, required=True, allow_blank=False,
                error_messages={"invalid_choice": "Not a recognized category."},
            )
        for col in NUM_COLS:
            self.fields[col] = serializers.FloatField(
                required=True,
                error_messages={"required": "This field is required.",
                                "invalid": "Enter a valid number."},
            )

    def validate(self, attrs):
        meta = get_field_metadata()
        errors: dict[str, list[str]] = {}
        for col in NUM_COLS:
            val = attrs.get(col)
            if val is None:
                continue
            if not math.isfinite(val):
                errors[col] = ["Value must be a finite number."]
                continue
            bounds = meta["numeric"].get(col)
            if bounds:
                span = max(bounds["max"] - bounds["min"], 1.0)
                lo = bounds["min"] - _RANGE_PAD * span
                hi = bounds["max"] + _RANGE_PAD * span
                if not (lo <= val <= hi):
                    errors[col] = [
                        f"Out of the expected range "
                        f"({bounds['min']:.2f} – {bounds['max']:.2f})."
                    ]
        if errors:
            raise serializers.ValidationError(errors)
        return attrs


def validate_batch_frame(df) -> tuple[list[str], dict]:
    """Structural validation for the batch CSV: missing columns + unknown categories.

    Returns (missing_columns, category_errors). Empty for a clean file.
    """
    from .schema import TRAIN_COLS

    missing = [c for c in TRAIN_COLS if c not in df.columns]
    if missing:
        return missing, {}

    meta = get_field_metadata()
    category_errors: dict[str, list[str]] = {}
    for col in CAT_COLS:
        allowed = set(meta["categorical"].get(col, []))
        seen = {str(v) for v in df[col].dropna().unique()}
        unknown = sorted(seen - allowed)
        if unknown:
            category_errors[col] = unknown[:10]
    return [], category_errors
