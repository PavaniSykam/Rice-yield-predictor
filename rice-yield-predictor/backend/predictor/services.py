"""
Inference services — wrap the existing trained pipeline. No retraining here.

Responsibilities: turn validated input into the exact DataFrame the pipeline expects,
run `model.predict`, and shape the yield / confidence / status payload the API returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model_loader import get_metrics, get_model
from .schema import TRAIN_COLS

# Yield bands (kept identical to the Streamlit app's thresholds).
_HIGH = 30_000
_MEDIUM = 15_000


def _confidence_half_width() -> float:
    """~95% band ≈ 2 × RMSE of the deployed model."""
    return 2.0 * get_metrics()["rmse"]


def classify(pred: float) -> tuple[str, str, str]:
    """Return (status label, status_class, recommendation)."""
    if pred >= _HIGH:
        return ("HIGH YIELD", "high",
                "Expected to exceed standard productivity benchmarks.")
    if pred >= _MEDIUM:
        return ("MEDIUM YIELD", "medium",
                "Consider optimizing fertilizer or irrigation timing.")
    return ("LOW YIELD", "low",
            "Review soil health, pest control and agronomic inputs.")


def _to_frame(records: list[dict]) -> pd.DataFrame:
    """Build a column-ordered DataFrame the pipeline can consume."""
    return pd.DataFrame(records)[TRAIN_COLS]


def predict_one(features: dict) -> dict:
    """Single prediction with confidence band, per-hectare yield and model metrics."""
    model = get_model()
    metrics = get_metrics()
    frame = _to_frame([features])
    pred = float(model.predict(frame)[0])

    half = _confidence_half_width()
    hectares = float(features.get("Hectares") or 0) or 1.0
    status, status_class, recommendation = classify(pred)

    return {
        "prediction": round(pred, 2),
        "yield_per_hectare": round(pred / hectares, 2),
        "status": status,
        "status_class": status_class,
        "recommendation": recommendation,
        "confidence": {
            "lower": round(max(0.0, pred - half), 2),
            "upper": round(pred + half, 2),
        },
        "model": {
            "name": metrics["best_model"],
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        },
    }


def predict_batch(df: pd.DataFrame) -> list[dict]:
    """Vectorised prediction over a validated DataFrame (batch CSV flow)."""
    model = get_model()
    half = _confidence_half_width()
    frame = df[TRAIN_COLS]
    preds = np.asarray(model.predict(frame), dtype=float)

    results = []
    for row, pred in zip(df.to_dict("records"), preds):
        status, status_class, _ = classify(float(pred))
        results.append({
            **row,
            "Predicted_Yield_Kg": round(float(pred), 2),
            "CI_Lower_Kg": round(max(0.0, float(pred) - half), 2),
            "CI_Upper_Kg": round(float(pred) + half, 2),
            "Status": status,
            "status_class": status_class,
        })
    return results
