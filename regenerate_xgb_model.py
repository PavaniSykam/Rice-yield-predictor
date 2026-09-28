"""
Regenerate the XGBoost artifacts so they load in THIS environment.
=================================================================
Why this is needed
    models/paddy_yield_model.pkl and models/XGBoost.pkl fail with
    `XGBoostError: input stream corrupted`. The pickle itself is fine —
    joblib unpickles it all the way down, but the *embedded XGBoost booster
    binary buffer* was written by a different XGBoost build than the one
    installed here, so XGBoosterUnserializeFromBuffer rejects it. (Typical
    when the .pkl was saved on Colab/Linux and loaded on Windows.)

Fix
    A buffer we cannot read cannot be re-serialized, so we retrain the *same*
    pipeline (identical preprocessing, SelectKBest, XGBoost + the notebook's
    GridSearch grid) locally and re-save it. The re-saved pickle is written by
    the local xgboost build, so the app loads it cleanly and "XGBoost (best)"
    reappears in the sidebar model picker.

    Originals are copied to models/_original_xgb_backup/ before overwriting.

Run
    python regenerate_xgb_model.py
"""

import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

# ── Load + clean exactly as the notebook / app do ──────────────────────────
df = pd.read_csv(BASE / "data" / "paddydataset.csv")
df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "_", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_")
    .str.replace("-", "_")
)
df = df.drop_duplicates()

# Columns the notebook dropped after EDA
DROP = ["LP_Mainfield_in_Tonnes", "LP_nurseryarea_in_Tonnes", "Nursery_area__Cents",
        "DAP_20days", "Weed28D_thiobencarb", "Micronutrients_70Days"]
df = df.drop(columns=[c for c in DROP if c in df.columns])
# ═══APPEND═══

y = df["Paddy_yield_in_Kg"]
X = df.drop(columns=["Paddy_yield_in_Kg"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Outlier clipping: limits fit on train, applied to both (as in the notebook)
X_tr, X_te = X_tr.copy(), X_te.copy()
for c in num_cols:
    q1, q3 = X_tr[c].quantile(0.25), X_tr[c].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    X_tr[c] = X_tr[c].clip(lo, hi)
    X_te[c] = X_te[c].clip(lo, hi)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
])
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_regression)),
    ("model", XGBRegressor(objective="reg:squarederror", random_state=42)),
])
param_grid = {
    "feature_selection__k": [10, 15, 20],
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.01, 0.1],
}

print("Training XGBoost pipeline (GridSearchCV, cv=5)...")
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_tr, y_tr)
best = grid.best_estimator_

y_pred = best.predict(X_te)
r2 = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred) ** 0.5
mae = mean_absolute_error(y_te, y_pred)
print(f"Best params : {grid.best_params_}")
print(f"Holdout     : R2={r2:.4f}  RMSE={rmse:,.1f}  MAE={mae:,.1f}")

# Back up the unreadable originals, then overwrite with the local build.
backup = MODELS / "_original_xgb_backup"
backup.mkdir(exist_ok=True)
for fname in ("paddy_yield_model.pkl", "XGBoost.pkl"):
    src = MODELS / fname
    if src.exists() and not (backup / fname).exists():
        shutil.copy2(src, backup / fname)
        print(f"Backed up {fname} -> {backup / fname}")

joblib.dump(best, MODELS / "paddy_yield_model.pkl")
joblib.dump(best, MODELS / "XGBoost.pkl")

# Verify the freshly written artifacts reload cleanly.
reloaded = joblib.load(MODELS / "paddy_yield_model.pkl")
_ = reloaded.predict(X_te.head(3))
print("Re-saved and verified: paddy_yield_model.pkl, XGBoost.pkl load OK.")

