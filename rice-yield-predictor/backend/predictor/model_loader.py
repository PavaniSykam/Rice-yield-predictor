"""
Model loader — loads the trained pipeline ONCE and caches it in memory.

The .pkl is never re-read per request. Keeping this isolated from the views makes
the inference path easy to test and swap. The metrics JSON is loaded the same way.
"""
from __future__ import annotations

import json
import threading

import joblib
from django.conf import settings

_model = None
_metrics: dict | None = None
_lock = threading.Lock()


def get_model():
    """Return the cached pipeline, loading it on first use (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = joblib.load(settings.MODEL_PATH)
    return _model


def get_metrics() -> dict:
    """Return the best model's holdout metrics (r2/rmse/mae) from model_metrics.json."""
    global _metrics
    if _metrics is None:
        with _lock:
            if _metrics is None:
                with open(settings.METRICS_PATH, encoding="utf-8") as f:
                    payload = json.load(f)
                best = payload.get("best_model")
                models = payload.get("models", {})
                perf = models.get(best, {})
                _metrics = {
                    "best_model": best,
                    "r2": round(float(perf.get("r2", 0.0)), 4),
                    "rmse": round(float(perf.get("rmse", 0.0)), 2),
                    "mae": round(float(perf.get("mae", 0.0)), 2),
                    "all_models": models,
                }
    return _metrics
