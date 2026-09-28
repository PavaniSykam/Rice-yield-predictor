"""
Feature schema for the trained pipeline.

These names are the EXACT columns the saved scikit-learn/XGBoost pipeline expects.
Do not rename them — the frontend uses human-readable labels, but the API contract
and the DataFrame handed to `model.predict()` must use these keys verbatim.
"""
from __future__ import annotations

import functools

import pandas as pd
from django.conf import settings

# Column order handed to the pipeline (target excluded).
TRAIN_COLS: list[str] = [
    "Hectares", "Agriblock", "Variety", "Soil_Types", "Seedrate_in_Kg",
    "Nursery", "Urea_40Days", "Potassh_50Days", "Pest_60Day_in_ml",
    "30DRain__in_mm", "30DAI_in_mm", "30_50DRain__in_mm", "30_50DAI_in_mm",
    "51_70DRain_in_mm", "51_70AI_in_mm", "71_105DRain_in_mm", "71_105DAI_in_mm",
    "Trash_in_bundles",
    "Min_temp_D1_D30", "Max_temp_D1_D30", "Min_temp_D31_D60", "Max_temp_D31_D60",
    "Min_temp_D61_D90", "Max_temp_D61_D90", "Min_temp_D91_D120", "Max_temp_D91_D120",
    "Inst_Wind_Speed_D1_D30_in_Knots", "Inst_Wind_Speed_D31_D60_in_Knots",
    "Inst_Wind_Speed_D61_D90_in_Knots", "Inst_Wind_Speed_D91_D120_in_Knots",
    "Wind_Direction_D1_D30", "Wind_Direction_D31_D60",
    "Wind_Direction_D61_D90", "Wind_Direction_D91_D120",
    "Relative_Humidity_D1_D30", "Relative_Humidity_D31_D60",
    "Relative_Humidity_D61_D90", "Relative_Humidity_D91_D120",
]

CAT_COLS: list[str] = [
    "Agriblock", "Variety", "Soil_Types", "Nursery",
    "Wind_Direction_D1_D30", "Wind_Direction_D31_D60",
    "Wind_Direction_D61_D90", "Wind_Direction_D91_D120",
]
NUM_COLS: list[str] = [c for c in TRAIN_COLS if c not in CAT_COLS]

TARGET = "Paddy_yield_in_Kg"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook/app cleaning (column normalisation + dedupe)."""
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "_", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    return df.drop_duplicates().reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    return _clean(pd.read_csv(settings.DATASET_PATH))


@functools.lru_cache(maxsize=1)
def get_field_metadata() -> dict:
    """Categorical options + numeric ranges/medians, derived from the dataset.

    Powers the frontend's dropdowns, slider bounds and sensible defaults so the
    UI stays in sync with the data the model was trained on.
    """
    df = _dataset()
    categorical = {
        c: sorted(str(v) for v in df[c].dropna().unique())
        for c in CAT_COLS if c in df.columns
    }
    numeric = {}
    for c in NUM_COLS:
        if c in df.columns:
            col = df[c]
            numeric[c] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "median": float(col.median()),
            }
    return {
        "train_cols": TRAIN_COLS,
        "categorical_cols": CAT_COLS,
        "numeric_cols": NUM_COLS,
        "categorical": categorical,
        "numeric": numeric,
    }
