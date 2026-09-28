"""
Train ALL paddy-yield models locally and (re)generate every artifact the app loads.
=====================================================================================
Why this exists
    The .pkl files shipped with the repo were trained on Colab/Linux. Loading them on
    a different scikit-learn / xgboost build risks version-mismatch warnings (and, for
    XGBoost, an outright "input stream corrupted" error). Training here, with the
    locally installed builds, guarantees every artifact loads cleanly and reproducibly.

What it does (mirrors ML_Paddy_pipeline(Regression).ipynb exactly)
    1. Load paddydataset.csv and normalise column names.
    2. Drop duplicate rows (451 of them) -- this MUST happen before the split so the
       app's holdout split (train_test_split(..., random_state=42) on the deduped
       frame) matches the set these models were evaluated on.
    3. Drop the six columns removed after EDA.
    4. train_test_split(test_size=0.2, random_state=42).
    5. Clip numeric outliers with IQR limits fit on the TRAIN split, applied to both
       splits (same as the notebook).
    6. Build Pipeline(preprocessor -> SelectKBest(f_regression) -> model) and tune each
       of the 9 regressors with GridSearchCV(cv=5, scoring="r2").
    7. Save models/<Name>.pkl for all 9, models/paddy_yield_model.pkl for the best,
       and models/model_metrics.json with this run's metrics + local library versions.

Run
    python train_models.py
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
TARGET = "Paddy_yield_in_Kg"

# Columns the notebook dropped after EDA (leakage / redundancy).
DROP_COLS = [
    "LP_Mainfield_in_Tonnes", "LP_nurseryarea_in_Tonnes", "Nursery_area__Cents",
    "DAP_20days", "Weed28D_thiobencarb", "Micronutrients_70Days",
]


def load_clean() -> pd.DataFrame:
    """Load the CSV and reproduce the notebook's cleaning up to the split."""
    df = pd.read_csv(BASE / "data" / "paddydataset.csv")
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "_", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    df = df.drop_duplicates()                       # 451 dupes -> must precede split
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def clip_outliers(X_tr: pd.DataFrame, X_te: pd.DataFrame, num_cols):
    """IQR clip: limits fit on TRAIN, applied to both splits (as in the notebook)."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in num_cols:
        q1, q3 = X_tr[c].quantile(0.25), X_tr[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X_tr[c] = X_tr[c].clip(lo, hi)
        X_te[c] = X_te[c].clip(lo, hi)
    return X_tr, X_te


def build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("model", estimator),
    ])


MODELS_TO_TRAIN = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
}

PARAM_GRIDS = {
    "LinearRegression": {"feature_selection__k": [10, 15, 20]},
    "Ridge": {"feature_selection__k": [10, 15, 20],
              "model__alpha": [0.01, 0.1, 1, 10]},
    "Lasso": {"feature_selection__k": [10, 15, 20],
              "model__alpha": [0.001, 0.01, 0.1, 1]},
    "DecisionTree": {"feature_selection__k": [10, 15, 20],
                     "model__max_depth": [5, 10, 20],
                     "model__min_samples_split": [2, 5, 10]},
    "RandomForest": {"feature_selection__k": [10, 15, 20],
                     "model__n_estimators": [100, 200],
                     "model__max_depth": [5, 10, 20]},
    "GradientBoosting": {"feature_selection__k": [10, 15, 20],
                         "model__n_estimators": [100, 200],
                         "model__learning_rate": [0.01, 0.1]},
    "KNN": {"feature_selection__k": [10, 15, 20],
            "model__n_neighbors": [3, 5, 7]},
    "SVR": {"feature_selection__k": [10, 15, 20],
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]},
    "XGBoost": {"feature_selection__k": [10, 15, 20],
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.01, 0.1]},
}


def main() -> None:
    df = load_clean()
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    X_tr, X_te = clip_outliers(X_tr, X_te, num_cols)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ])

    MODELS.mkdir(exist_ok=True)
    best_estimators, metrics_table = {}, {}

    for name, estimator in MODELS_TO_TRAIN.items():
        print(f"Training {name} ...")
        grid = GridSearchCV(
            build_pipeline(preprocessor, estimator),
            PARAM_GRIDS[name], cv=5, scoring="r2", n_jobs=-1,
        )
        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_
        best_estimators[name] = best

        y_pred = best.predict(X_te)
        metrics_table[name] = {
            "r2": float(r2_score(y_te, y_pred)),
            "train_r2": float(r2_score(y_tr, best.predict(X_tr))),
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "mae": float(mean_absolute_error(y_te, y_pred)),
        }
        print(f"    best params : {grid.best_params_}")
        print(f"    holdout     : R2={metrics_table[name]['r2']:.4f}  "
              f"RMSE={metrics_table[name]['rmse']:,.1f}  "
              f"MAE={metrics_table[name]['mae']:,.1f}")

    best_name = max(metrics_table, key=lambda n: metrics_table[n]["r2"])
    print(f"\nBest model: {best_name} (R2={metrics_table[best_name]['r2']:.4f})")

    # Back up existing artifacts once, then overwrite with the local build.
    backup = MODELS / "_original_backup"
    for name in list(MODELS_TO_TRAIN) + ["paddy_yield_model"]:
        src = MODELS / f"{name}.pkl"
        if src.exists():
            backup.mkdir(exist_ok=True)
            dst = backup / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    for name, est in best_estimators.items():
        joblib.dump(est, MODELS / f"{name}.pkl")
    joblib.dump(best_estimators[best_name], MODELS / "paddy_yield_model.pkl")

    payload = {
        "best_model": best_name,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "models": metrics_table,
    }
    with open(MODELS / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Verify every freshly written artifact reloads cleanly in this environment.
    for name in list(MODELS_TO_TRAIN) + ["paddy_yield_model"]:
        reloaded = joblib.load(MODELS / f"{name}.pkl")
        _ = reloaded.predict(X_te.head(3))
    print(f"\nSaved 9 models + paddy_yield_model.pkl + model_metrics.json "
          f"(scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}).")
    print("All artifacts reload and predict cleanly.")


if __name__ == "__main__":
    main()

