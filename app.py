"""
Rice Yield Predictor — Agricultural Intelligence Through Data Science
=====================================================================
Light-themed Streamlit app for paddy yield prediction.

Navigation : left sidebar (Predict Yield | Model Info | Guide)
Model      : single saved pipeline — models/paddy_yield_model.pkl
             (preprocessor -> SelectKBest -> XGBoost, best of 9 tuned regressors)

Visual language (mirrors the reference "Element Explorer" dashboard)
    - light neutral page background, white content cards
    - thin subtle borders, restrained rounded corners, minimal shadow
    - dark navy/charcoal typography, single restrained green accent
Layout (Predict page)
    Desktop : two-column input/output layout with a sticky result panel.
    Tablet  : input and result panels stack vertically; compact grids stay 2-up.
    Phone   : cards and form fields collapse to one column with roomy spacing.
    Batch mode swaps the two panels for an upload card + results table.
"""

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Rice Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Design tokens (mirrored in CSS :root below so Altair matches the DOM)
# ══════════════════════════════════════════════════════════════════════════════
INK = "#1A2233"        # primary navy/charcoal text
INK_SOFT = "#5B6472"   # secondary muted text
ACCENT = "#2E7D5B"     # restrained forest-green accent
ACCENT_SOFT = "#E7F1EC"
GRID = "#E7EAEE"        # thin chart gridlines / borders
SUCCESS = "#2E7D5B"
WARNING = "#B7791F"
ERROR = "#C0492F"

# ══════════════════════════════════════════════════════════════════════════════
# Custom light theme CSS
# ══════════════════════════════════════════════════════════════════════════════
LIGHT_THEME_CSS = """
<style>
    :root {
        --bg-primary: #F5F6F8;
        --card-bg: #FFFFFF;
        --sidebar-bg: #FFFFFF;
        --text-primary: #1A2233;
        --text-secondary: #5B6472;
        --border-color: #E3E6EA;
        --border-strong: #D5DAE1;
        --accent: #2E7D5B;
        --accent-soft: #E7F1EC;
        --success: #2E7D5B;
        --warning: #B7791F;
        --error: #C0492F;
        --radius: 10px;
        --radius-sm: 8px;
        --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.05);
        --shadow-md: 0 1px 3px rgba(16, 24, 40, 0.08);
    }

    /* ── Base ── */
    .stApp { background-color: var(--bg-primary); }
    .stApp, .stApp p, .stApp li { color: var(--text-primary); }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1360px;
        padding: 2rem 1.4rem 2.5rem;
    }

    /* ── Headings ── */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text-primary);
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        letter-spacing: -0.01em;
    }
    .stApp h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.15rem; }
    .stApp h2 { font-size: 1.35rem; font-weight: 650; }
    .stApp h3 { font-size: 1.1rem; font-weight: 600; }
    .stApp h4 { font-size: 0.98rem; font-weight: 600; color: var(--text-primary); }

    /* ── Secondary text (labels + captions) ── */
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCaptionContainer"] {
        color: var(--text-secondary);
        font-size: 0.82rem;
    }
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stCaptionContainer"] { overflow-wrap: anywhere; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    .brand {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.2rem 0.25rem 0.9rem;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-color);
    }
    .brand-mark {
        width: 38px; height: 38px; border-radius: 10px;
        background: var(--accent-soft);
        display: grid; place-items: center; font-size: 1.3rem;
    }
    .brand-name { font-weight: 700; font-size: 1.05rem; line-height: 1.1; color: var(--text-primary); }
    .brand-sub { font-size: 0.72rem; color: var(--text-secondary); }
    .nav-label {
        text-transform: uppercase; letter-spacing: 0.08em;
        font-size: 0.7rem; font-weight: 600; color: var(--text-secondary);
        margin: 0.9rem 0.25rem 0.35rem;
    }

    /* Sidebar radio rendered as a clean vertical nav */
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 0.2rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.5rem 0.6rem; border-radius: var(--radius-sm);
        width: 100%; cursor: pointer; transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #F1F3F5; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--accent-soft);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: var(--accent); font-weight: 600;
    }
    [data-testid="stSidebar"] [role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.92rem;
    }
    /* Hide the default radio dot so items read as nav rows */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }

    /* ── Tabs (used inside the weather card) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] { color: var(--text-secondary); }
    .stApp .stTabs [data-baseweb="tab"] p { color: inherit; font-size: 0.88rem; }
    .stTabs [aria-selected="true"] { color: var(--accent); font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

    /* ── Inputs (number, select, slider) ── */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: var(--border-strong);
        border-radius: var(--radius-sm);
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(46, 125, 91, 0.12);
    }
    [data-testid="stSlider"] [role="slider"] { background-color: var(--accent); }

    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {
        background-color: var(--accent);
        color: #FFFFFF;
        border: 1px solid var(--accent);
        border-radius: var(--radius-sm);
        font-weight: 600;
        padding: 0.45rem 1.1rem;
        width: 100%;
        box-shadow: none;
        transition: background-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stApp .stButton > button p,
    .stApp .stDownloadButton > button p { color: inherit; }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #276A4D;
        border-color: #276A4D;
        color: #FFFFFF;
    }
    /* Secondary (reset) button reads as an outline control */
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF;
        color: var(--text-primary);
        border: 1px solid var(--border-strong);
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #F1F3F5;
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    /* ── Cards (st.container(border=True)) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.15rem; }

    .section-title {
        color: var(--text-primary);
        font-weight: 650;
        font-size: 1.02rem;
        margin-bottom: 0.1rem;
    }
    .section-sub {
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-bottom: 0.85rem;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 0.8rem 1rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-color);
    }
    .stApp [data-testid="stMetricLabel"] p { color: var(--text-secondary); font-size: 0.8rem; }
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] * {
        color: var(--text-primary); font-weight: 700;
    }
    .stApp [data-testid="stMetricValue"] { font-size: 1.35rem; }
    [data-testid="stMetricValue"] > div {
        overflow: visible; text-overflow: clip; white-space: normal;
    }

    /* ── Application hero ── */
    .app-hero { margin: 0 0 1.1rem; }
    .app-hero h1 { margin-bottom: 0.2rem; }
    .app-hero p {
        margin: 0; color: var(--text-secondary);
        font-size: 0.92rem; font-weight: 500;
    }

    /* ── Result card (light, minimal shadow, accent top rule) ── */
    .result-card {
        background: #FFFFFF;
        padding: 1.5rem 1.2rem;
        border-radius: var(--radius);
        border: 1px solid var(--border-color);
        border-top: 3px solid var(--accent);
        text-align: center;
        box-shadow: var(--shadow-md);
        margin-bottom: 1rem;
    }
    .result-value {
        color: var(--text-primary);
        font-size: 2.4rem; font-weight: 700;
        margin: 0.35rem 0; line-height: 1.1;
    }
    .result-value span { color: var(--text-secondary); }
    .result-label {
        color: var(--text-secondary); font-size: 0.78rem;
        letter-spacing: 0.08em; text-transform: uppercase;
    }

    /* ── Status badges ── */
    .status-badge {
        padding: 0.32rem 0.9rem; border-radius: 999px;
        font-weight: 600; font-size: 0.8rem;
        display: inline-block; margin-top: 0.55rem;
    }
    .status-high   { background: rgba(46,125,91,.12);  color: var(--success); border: 1px solid rgba(46,125,91,.35); }
    .status-medium { background: rgba(183,121,31,.12); color: var(--warning); border: 1px solid rgba(183,121,31,.35); }
    .status-low    { background: rgba(192,73,47,.10);  color: var(--error);   border: 1px solid rgba(192,73,47,.30); }

    /* ── Confidence band ── */
    .confidence-band {
        font-size: .82rem; color: var(--text-secondary);
        background: #F5F6F8; border: 1px solid var(--border-color);
        border-radius: var(--radius-sm); display: block;
        padding: .5rem; margin-top: 0.85rem; text-align: center;
    }
    .confidence-band b { color: var(--text-primary); }

    /* ── File uploader dropzone ── */
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed var(--border-strong);
        border-radius: var(--radius); background: #FBFCFD;
    }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] { border-radius: var(--radius-sm); }

    /* Altair charts never exceed their card width. */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        width: 100% !important; max-width: 100%; overflow-x: auto;
    }

    /* ── Responsive layout system ── */
    .sticky-anchor { height: 0; }

    @media (min-width: 1101px) {
        div[data-testid="stColumn"]:has(.sticky-anchor),
        div[data-testid="column"]:has(.sticky-anchor) {
            position: sticky; top: 4.5rem; align-self: flex-start;
        }
    }

    /* Predictor split becomes a vertical flow on tablets. */
    @media (max-width: 1100px) {
        div[data-testid="stHorizontalBlock"]:has(.sticky-anchor) > div[data-testid="stColumn"],
        div[data-testid="stHorizontalBlock"]:has(.sticky-anchor) > div[data-testid="column"] {
            flex: 1 1 100% !important; width: 100% !important; max-width: 100% !important;
        }
    }

    @media (max-width: 768px) {
        .block-container,
        [data-testid="stMainBlockContainer"] { max-width: 100%; padding: 1.2rem 0.85rem 1.5rem; }
        .stApp h1 { font-size: clamp(1.55rem, 6vw, 1.9rem); line-height: 1.1; }
        .stApp h2 { font-size: 1.2rem; }
        .result-value { font-size: clamp(1.9rem, 9vw, 2.3rem); }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem; overflow-x: auto; scrollbar-width: none; padding-bottom: 0.15rem;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] { flex: 0 0 auto; padding: 0.5rem 0.7rem; white-space: nowrap; }

        div[data-testid="stHorizontalBlock"] { gap: 0.7rem !important; flex-wrap: wrap !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 calc(50% - 0.35rem) !important;
            width: calc(50% - 0.35rem) !important;
            max-width: calc(50% - 0.35rem) !important; min-width: 0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.sticky-anchor) > div[data-testid="stColumn"],
        div[data-testid="stHorizontalBlock"]:has(.sticky-anchor) > div[data-testid="column"] {
            flex-basis: 100% !important; width: 100% !important; max-width: 100% !important;
        }
        [data-testid="stMetric"] { min-height: 72px; padding: 0.65rem 0.8rem; }
        .stButton > button, .stDownloadButton > button { min-height: 2.7rem; }
    }

    /* Narrow phones: one field per row. */
    @media (max-width: 520px) {
        .block-container,
        [data-testid="stMainBlockContainer"] { padding-left: 0.7rem; padding-right: 0.7rem; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex-basis: 100% !important; width: 100% !important; max-width: 100% !important;
        }
        .result-value { font-size: 1.9rem; }
        [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #C7CDD4; border-radius: 4px; }
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Feature schema — EXACT column names expected by the trained pipeline
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_COLS = [
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
CAT_COLS = ["Agriblock", "Variety", "Soil_Types", "Nursery",
            "Wind_Direction_D1_D30", "Wind_Direction_D31_D60",
            "Wind_Direction_D61_D90", "Wind_Direction_D91_D120"]
NUM_COLS = [c for c in TRAIN_COLS if c not in CAT_COLS]

INTERVALS = ["D1_D30", "D31_D60", "D61_D90", "D91_D120"]
INTERVAL_NAMES = ["Days 1–30", "Days 31–60", "Days 61–90", "Days 91–120"]
INTERVAL_TIPS = {
    "D1_D30": "Establishment phase — requires steady moisture.",
    "D31_D60": "Tillering phase — sets the overall yield ceiling.",
    "D61_D90": "Flowering phase — critical weather sensitivity.",
    "D91_D120": "Grain filling — dry, sunny days favor heavy grain.",
}
RAIN_COLS = {
    "D1_D30": ["30DRain__in_mm", "30DAI_in_mm"],
    "D31_D60": ["30_50DRain__in_mm", "30_50DAI_in_mm"],
    "D61_D90": ["51_70DRain_in_mm", "51_70AI_in_mm"],
    "D91_D120": ["71_105DRain_in_mm", "71_105DAI_in_mm"],
}

# MODEL_PERF, CONFIDENCE_HALF_WIDTH and ALL_MODEL_RESULTS are derived from
# models/model_metrics.json in the loaders section below and refreshed whenever
# the user switches the active model in the sidebar.

# ══════════════════════════════════════════════════════════════════════════════
# Data & model loaders (paths resolved relative to this file)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent


def _find(name: str, subdir: str) -> Path:
    """Locate a file whether it sits in a subfolder or beside the script."""
    for candidate in (BASE_DIR / subdir / name, BASE_DIR / name, Path(name)):
        if candidate.exists():
            return candidate
    return BASE_DIR / subdir / name  # default; will raise a clear error on load


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(_find("paddydataset.csv", "data"))
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "_", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    # Drop duplicates BEFORE anything else so the holdout split reproduced in
    # holdout_predictions() (train_test_split, random_state=42) matches the exact
    # test set the models were trained/evaluated against. Without this the app
    # split over the 451 duplicate rows, leaking train rows into the "holdout".
    df = df.drop_duplicates().reset_index(drop=True)
    return df



@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(_find("model_metrics.json", "models"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_plot_manifest() -> dict:
    """Chart assets produced by ML_Paddy_train_models.ipynb (assets/plots/).
    Returns {} when the notebook hasn't been run, so the app still works."""
    try:
        with open(_find("manifest.json", "assets/plots"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}



@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model(path: str):
    """Load a saved pipeline by absolute path (cached per path)."""
    return joblib.load(path)


# (display name, pickle filename, model_metrics.json key)
# paddy_yield_model.pkl is the deployed 'best' pipeline (XGBoost).
MODEL_REGISTRY = [
    ("XGBoost (best)",    "paddy_yield_model.pkl", "XGBoost"),
    ("Random Forest",     "RandomForest.pkl",      "RandomForest"),
    ("Gradient Boosting", "GradientBoosting.pkl",  "GradientBoosting"),
    ("Decision Tree",     "DecisionTree.pkl",      "DecisionTree"),
    ("KNN",               "KNN.pkl",               "KNN"),
    ("Ridge",             "Ridge.pkl",             "Ridge"),
    ("Lasso",             "Lasso.pkl",             "Lasso"),
    ("Linear Regression", "LinearRegression.pkl",  "LinearRegression"),
    ("SVR",               "SVR.pkl",               "SVR"),
]
DISPLAY_FOR_KEY = {key: disp for disp, _fname, key in MODEL_REGISTRY}

# --- Dataset ---------------------------------------------------------------
data_error = ""
try:
    df = load_data()
    data_loaded = True
except Exception as e:  # noqa: BLE001 - surfaced to the user in the UI
    df = pd.DataFrame()
    data_loaded = False
    data_error = f"{type(e).__name__}: {e}"

# --- Metrics ---------------------------------------------------------------
try:
    METRICS = load_metrics().get("models", {})
except Exception:  # noqa: BLE001
    METRICS = {}

# --- Saved chart assets (from the training notebook) -----------------------
PLOT_MANIFEST = load_plot_manifest()

# --- Probe which model artifacts actually load in this environment ---------
AVAILABLE_MODELS = []      # list of (display, path, key) that loaded cleanly
MODEL_LOAD_ERRORS = {}     # display -> error string for artifacts that failed
for disp, fname, key in MODEL_REGISTRY:
    path = str(_find(fname, "models"))
    try:
        load_model(path)
        AVAILABLE_MODELS.append((disp, path, key))
    except Exception as e:  # noqa: BLE001
        MODEL_LOAD_ERRORS[disp] = f"{type(e).__name__}: {e}"

model_loaded = data_loaded and bool(AVAILABLE_MODELS)
load_error = data_error or ("No model artifacts could be loaded." if not AVAILABLE_MODELS else "")


def perf_for(key: str) -> dict:
    m = METRICS.get(key, {})
    return {"r2": m.get("r2", float("nan")),
            "rmse": m.get("rmse", float("nan")),
            "mae": m.get("mae", float("nan"))}


# Benchmark table (all candidates present in the metrics file)
if METRICS:
    ALL_MODEL_RESULTS = (
        pd.DataFrame([
            {"Model": DISPLAY_FOR_KEY.get(k, k),
             "Test R²": round(v.get("r2", float("nan")), 4),
             "RMSE (Kg)": round(v.get("rmse", 0)),
             "MAE (Kg)": round(v.get("mae", 0))}
            for k, v in METRICS.items()
        ])
        .sort_values("Test R²", ascending=False)
        .reset_index(drop=True)
    )
else:
    ALL_MODEL_RESULTS = pd.DataFrame(columns=["Model", "Test R²", "RMSE (Kg)", "MAE (Kg)"])

# Defaults; overridden once the user picks a model in the sidebar.
model = None
SELECTED_NAME = ""
SELECTED_KEY = ""
MODEL_PERF = {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
CONFIDENCE_HALF_WIDTH = 0.0

if data_loaded:
    NUM_RANGES = {c: (float(df[c].min()), float(df[c].max())) for c in NUM_COLS}
    HECTARE_OPTIONS = sorted(int(v) for v in df["Hectares"].unique())
    AGRIBLOCK_OPTS = sorted(df["Agriblock"].dropna().unique())
    VARIETY_OPTS = sorted(df["Variety"].dropna().unique())
    SOIL_OPTS = sorted(df["Soil_Types"].dropna().unique())
    NURSERY_OPTS = sorted(df["Nursery"].dropna().unique())
    WIND_OPTS = {i: sorted(df[f"Wind_Direction_{i}"].dropna().unique()) for i in INTERVALS}

# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def ui_range(col: str, pad_frac: float = 0.15):
    lo, hi = NUM_RANGES[col]
    span = max(hi - lo, 1.0)
    return max(lo - pad_frac * span, 0.0), hi + pad_frac * span


def section_title(icon: str, title: str, sub: str = "") -> None:
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="section-title">{icon}&nbsp;{title}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def num_input(target, label: str, col: str, step: float, key: str, help: str | None = None):
    """Number input bounded by the dataset range and defaulting to the median."""
    lo, hi = ui_range(col)
    return target.number_input(label, lo, hi, float(df[col].median()),
                               step=step, key=key, help=help)


def slider_input(target, label: str, col: str, step: float, key: str):
    lo, hi = ui_range(col)
    return target.slider(label, lo, hi, float(df[col].median()), step=step, key=key)


def category_for(pred: float):
    if pred >= 30_000:
        return "status-high", "HIGH YIELD", "Expected to exceed standard productivity benchmarks."
    if pred >= 15_000:
        return "status-medium", "MEDIUM YIELD", "Consider optimizing fertilizer or irrigation timing."
    return "status-low", "LOW YIELD", "Review soil health, pest control and agronomic inputs."


def reset_inputs():
    for key in list(st.session_state.keys()):
        if key.startswith(("in_", "sel_")):
            del st.session_state[key]
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Visualization helpers (light theme)
# ══════════════════════════════════════════════════════════════════════════════
def apply_light(chart, title: str = ""):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID, labelColor=INK_SOFT, titleColor=INK,
            labelFontSize=10, titleFontSize=11, domain=False, tickColor=GRID,
        )
        .configure_legend(labelColor=INK_SOFT, titleColor=INK)
        .configure_title(color=INK, fontSize=13, anchor="start", fontWeight=600)
        .properties(title=title, background="rgba(0,0,0,0)")
    )


@st.cache_data(show_spinner=False)
def feature_importance_data(model_key: str, _model) -> pd.DataFrame | None:
    """Tree models expose feature_importances_; linear models expose coef_.
    Inputs are standardized by the pipeline, so |coef| is comparable across
    features. Distance/kernel models (KNN, rbf-SVR) expose neither -> None.
    `model_key` is hashed for caching (`_model` is skipped by Streamlit)."""
    try:
        names_all = _model.named_steps["preprocessor"].get_feature_names_out()
        sel = _model.named_steps["feature_selection"].get_support(indices=True)
        est = _model.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            scores = est.feature_importances_
        elif hasattr(est, "coef_"):
            import numpy as np
            scores = np.abs(est.coef_).ravel()
        else:
            return None
        return pd.DataFrame({
            "Feature": [n.split("__", 1)[-1] for n in names_all[sel]],
            "Importance": scores,
        })
    except Exception:
        return None


def feature_importance_chart():
    fdf = feature_importance_data(SELECTED_NAME, model)
    if fdf is None or fdf.empty:
        return None
    top_fdf = fdf.sort_values("Importance", ascending=False).head(15)
    ch = (
        alt.Chart(top_fdf)
        .mark_bar(cornerRadiusEnd=3, color=ACCENT)
        .encode(
            x=alt.X("Importance:Q", title="Importance score"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(height=360)
    )
    return apply_light(ch, "Top 15 most important features")

@st.cache_data(show_spinner="Computing holdout metrics...")
def holdout_predictions(model_key: str, _model) -> pd.DataFrame | None:
    try:
        X = df[TRAIN_COLS]
        y = df["Paddy_yield_in_Kg"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        X_te = X_te.copy()
        for c in NUM_COLS:
            q1, q3 = X_tr[c].quantile(0.25), X_tr[c].quantile(0.75)
            iqr = q3 - q1
            X_te[c] = X_te[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        y_pred = _model.predict(X_te)
        return pd.DataFrame({"Actual": y_te.values, "Predicted": y_pred})
    except Exception:
        return None


def residuals_chart(res: pd.DataFrame):
    res = res.assign(Residual=res["Actual"] - res["Predicted"])
    points = (
        alt.Chart(res)
        .mark_circle(size=42, opacity=0.55, color=ACCENT)
        .encode(
            x=alt.X("Predicted:Q", title="Predicted yield (Kg)"),
            y=alt.Y("Residual:Q", title="Residual error (Kg)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Actual:Q", format=",.0f"),
                     alt.Tooltip("Predicted:Q", format=",.0f"),
                     alt.Tooltip("Residual:Q", format=",.0f")],
        )
    )
    zero_line = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
        color=INK_SOFT, strokeDash=[4, 4]).encode(y="v:Q")
    return apply_light((points + zero_line).properties(height=320),
                       "Model residuals on holdout data")

# ══════════════════════════════════════════════════════════════════════════════
# Predict-page building blocks
# ══════════════════════════════════════════════════════════════════════════════
def farm_profile_card() -> dict:
    """Land, crop and agronomic inputs — one bordered card, responsive grid."""
    v = {}
    with st.container(border=True):
        section_title("🌱", "Farm profile", "Land, crop variety and applied inputs")

        c1, c2, c3 = st.columns(3)
        v["Hectares"] = c1.selectbox("Hectares", HECTARE_OPTIONS, key="sel_Hectares",
                                     help="Total area cultivated.")
        v["Agriblock"] = c2.selectbox("Agriblock", AGRIBLOCK_OPTS, key="sel_Agriblock")
        v["Variety"] = c3.selectbox("Paddy variety", VARIETY_OPTS, key="sel_Variety")

        c4, c5, c6 = st.columns(3)
        v["Soil_Types"] = c4.selectbox("Soil type", SOIL_OPTS, key="sel_Soil_Types")
        v["Nursery"] = c5.selectbox("Nursery method", NURSERY_OPTS, key="sel_Nursery")
        v["Seedrate_in_Kg"] = num_input(c6, "Seed rate (Kg)", "Seedrate_in_Kg", 10.0, "in_Seedrate")

        c7, c8, c9, c10 = st.columns(4)
        v["Urea_40Days"] = num_input(c7, "Urea @ 40 days (Kg)", "Urea_40Days", 5.0, "in_Urea")
        v["Potassh_50Days"] = num_input(c8, "Potash @ 50 days (Kg)", "Potassh_50Days", 5.0, "in_Potash")
        v["Pest_60Day_in_ml"] = num_input(c9, "Pesticide @ 60 days (ml)", "Pest_60Day_in_ml", 10.0, "in_Pest")
        v["Trash_in_bundles"] = num_input(c10, "Trash (bundles)", "Trash_in_bundles", 10.0, "in_Trash")
    return v


def weather_section() -> dict:
    """One tab per growth stage; inputs laid out in aligned rows (2 / 2 / 3)."""
    v = {}
    with st.container(border=True):
        section_title("🌦️", "Seasonal weather", "Conditions across the four crop-growth stages")
        stage_tabs = st.tabs(INTERVAL_NAMES)

        for tab, intv in zip(stage_tabs, INTERVALS):
            with tab:
                st.caption(INTERVAL_TIPS[intv])
                rain_col, irr_col = RAIN_COLS[intv]

                r1, r2 = st.columns(2)
                v[rain_col] = num_input(r1, "Rainfall (mm)", rain_col, 1.0, f"in_{rain_col}")
                v[irr_col] = num_input(r2, "Irrigation (mm)", irr_col, 1.0, f"in_{irr_col}")

                t1, t2 = st.columns(2)
                v[f"Min_temp_{intv}"] = slider_input(
                    t1, "Min temp (°C)", f"Min_temp_{intv}", 0.5, f"in_min_{intv}")
                v[f"Max_temp_{intv}"] = slider_input(
                    t2, "Max temp (°C)", f"Max_temp_{intv}", 0.5, f"in_max_{intv}")

                w1, w2, w3 = st.columns(3)
                ws_col = f"Inst_Wind_Speed_{intv}_in_Knots"
                v[ws_col] = slider_input(w1, "Wind speed (knots)", ws_col, 0.5, f"in_ws_{intv}")
                v[f"Wind_Direction_{intv}"] = w2.selectbox(
                    "Wind direction", WIND_OPTS[intv], key=f"sel_wd_{intv}")
                v[f"Relative_Humidity_{intv}"] = slider_input(
                    w3, "Humidity (%)", f"Relative_Humidity_{intv}", 1.0, f"in_rh_{intv}")
    return v

def render_result(features: dict) -> None:
    """Right-hand panel. The anchor div makes this column sticky (see CSS)."""
    st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)
    section_title("📊", "Yield output", "Updates live as you change inputs")

    sample_df = pd.DataFrame([features])[TRAIN_COLS]
    pred_val = float(model.predict(sample_df)[0])
    badge_cls, status_lbl, advice_txt = category_for(pred_val)
    ci_low = max(0.0, pred_val - CONFIDENCE_HALF_WIDTH)
    ci_high = pred_val + CONFIDENCE_HALF_WIDTH

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted paddy yield</div>
            <div class="result-value">{pred_val:,.0f} <span style="font-size: 1.1rem;">Kg</span></div>
            <div class="{badge_cls} status-badge">{status_lbl}</div>
            <div class="confidence-band">
                95% band: <b>{ci_low:,.0f}</b> – <b>{ci_high:,.0f}</b> Kg
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(advice_txt)

    m1, m2 = st.columns(2)
    m1.metric("Yield / ha", f"{pred_val / max(features['Hectares'], 1):,.0f} Kg")
    m2.metric("Model R²", f"{MODEL_PERF['r2']:.4f}")


def render_single() -> None:
    main_col, result_col = st.columns([66, 34], gap="large")
    with main_col:
        inputs = farm_profile_card()
        inputs.update(weather_section())
    with result_col:
        render_result(inputs)

def render_batch() -> None:
    left, right = st.columns([60, 40], gap="large")

    with left:
        with st.container(border=True):
            section_title("📁", "Batch predictions", "Upload a CSV with one row per field")
            uploaded = st.file_uploader("Field conditions CSV", type=["csv"],
                                        label_visibility="collapsed")
            if uploaded is not None:
                try:
                    batch_df = pd.read_csv(uploaded)
                    missing = [c for c in TRAIN_COLS if c not in batch_df.columns]
                    if missing:
                        st.error(f"Missing {len(missing)} required columns, e.g. "
                                 f"{', '.join(missing[:5])} …")
                    else:
                        st.success(f"CSV validated: {len(batch_df):,} rows detected.")
                        if st.button("Process batch predictions", type="primary"):
                            preds = model.predict(batch_df[TRAIN_COLS]).astype(float)
                            out = batch_df.copy()
                            out["Predicted_Yield_Kg"] = preds.round(0)
                            out["CI_Lower_Kg"] = (preds - CONFIDENCE_HALF_WIDTH).clip(min=0).round(0)
                            out["CI_Upper_Kg"] = (preds + CONFIDENCE_HALF_WIDTH).round(0)
                            st.session_state["batch_results"] = out
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error reading file: {exc}")

    with right:
        with st.container(border=True):
            section_title("🧾", "Required columns", f"{len(TRAIN_COLS)} training features, exact names")
            with st.expander("Show column list"):
                st.caption(", ".join(f"`{c}`" for c in TRAIN_COLS))

    if "batch_results" in st.session_state:
        out = st.session_state["batch_results"]
        st.markdown("#### Batch summary")
        b1, b2, b3 = st.columns(3)
        b1.metric("Total fields", f"{len(out):,}")
        b2.metric("Mean yield", f"{out['Predicted_Yield_Kg'].mean():,.0f} Kg")
        b3.metric("Max yield", f"{out['Predicted_Yield_Kg'].max():,.0f} Kg")

        st.dataframe(
            out[["Predicted_Yield_Kg", "CI_Lower_Kg", "CI_Upper_Kg"] + TRAIN_COLS[:3]],
            width="stretch", height=280,
        )
        st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"),
                           file_name="yield_predictions.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════
def page_predict(batch_mode: bool) -> None:
    st.markdown(
        '<div class="app-hero"><h1>Predict yield</h1>'
        '<p>Enter field conditions to estimate paddy yield in real time.</p></div>',
        unsafe_allow_html=True,
    )
    if batch_mode:
        render_batch()
    else:
        render_single()


def page_model_info() -> None:
    st.markdown(
        f'<div class="app-hero"><h1>Model info</h1>'
        f'<p>Active model: <b>{SELECTED_NAME}</b> · pipeline: preprocessor → SelectKBest → '
        f'estimator (best of 9 tuned regressors).</p></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² score", f"{MODEL_PERF['r2']:.4f}")
    k2.metric("RMSE", f"{MODEL_PERF['rmse']:.1f} Kg")
    k3.metric("MAE", f"{MODEL_PERF['mae']:.1f} Kg")
    k4.metric("95% band", f"± {CONFIDENCE_HALF_WIDTH:,.0f} Kg")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    col_tbl, col_imp = st.columns([42, 58], gap="large")
    with col_tbl:
        with st.container(border=True):
            section_title("🏆", "Candidate benchmark",
                          f"{len(ALL_MODEL_RESULTS)} tuned regressors on the holdout set")
            st.dataframe(
                ALL_MODEL_RESULTS.style
                    .highlight_max(subset=["Test R²"], color=ACCENT_SOFT)
                    .highlight_min(subset=["RMSE (Kg)"], color=ACCENT_SOFT),
                width="stretch", hide_index=True,
                height=len(ALL_MODEL_RESULTS) * 35 + 38,
            )
    with col_imp:
        with st.container(border=True):
            section_title("📶", "Feature importance",
                          f"Top drivers for {SELECTED_NAME}")
            chart_imp = feature_importance_chart()
            if chart_imp is not None:
                st.altair_chart(chart_imp, width="stretch")
            else:
                st.info(f"{SELECTED_NAME} does not expose per-feature importances "
                        "(e.g. KNN / SVR). Pick a tree or linear model to see this chart.")

    holdout_df = holdout_predictions(SELECTED_NAME, model)
    if holdout_df is not None:
        with st.container(border=True):
            section_title("📉", "Residual analysis",
                          f"{SELECTED_NAME} prediction error across the holdout sample")
            st.altair_chart(residuals_chart(holdout_df), width="stretch")

    # --- Saved training diagnostics (assets/plots via the notebook) --------
    if PLOT_MANIFEST:
        info = PLOT_MANIFEST.get("models", {}).get(SELECTED_KEY)
        diag = BASE_DIR / info["plot"] if info else None
        if diag and diag.exists():
            with st.container(border=True):
                section_title("🖼️", "Training diagnostics",
                              f"Actual vs predicted, residuals and feature drivers for {SELECTED_NAME}")
                st.image(str(diag), width="stretch")
                st.caption("Generated by ML_Paddy_train_models.ipynb on the holdout split.")

        comp = PLOT_MANIFEST.get("comparison_plot")
        comp_path = BASE_DIR / comp if comp else None
        if comp_path and comp_path.exists():
            with st.container(border=True):
                section_title("📊", "Model comparison",
                              "All 9 tuned pipelines — test R², RMSE and MAE")
                st.image(str(comp_path), width="stretch")

    if MODEL_LOAD_ERRORS:
        with st.container(border=True):
            section_title("🛠️", "Unavailable models",
                          "These artifacts could not be loaded in this environment")
            for name, err in MODEL_LOAD_ERRORS.items():
                st.markdown(f"- **{name}** — `{err}`")
            st.caption(
                "XGBoost pickles fail with *input stream corrupted* when the saved booster "
                "buffer was written by a different XGBoost build than the one installed here. "
                "Fix: re-save the model in this environment, or export the booster with "
                "`booster.save_model('model.json')` (the JSON/UBJ format is version-portable) "
                "and rebuild the pipeline around it."
            )

def page_guide() -> None:
    st.markdown(
        '<div class="app-hero"><h1>Guide</h1>'
        '<p>How to use the predictor and prepare batch files.</p></div>',
        unsafe_allow_html=True,
    )
    g1, g2 = st.columns(2, gap="large")
    with g1:
        with st.container(border=True):
            st.markdown(
                """
                #### How to use the predictor
                1. **Farm profile** — choose field size, agriblock, variety and soil, then enter
                   seed, fertilizer and pesticide inputs.
                2. **Seasonal weather** — use the stage tabs to enter conditions for each 30-day period:
                   - **Days 1–30**: establishment and early root development.
                   - **Days 31–60**: active tillering.
                   - **Days 61–90**: flowering and panicle initiation.
                   - **Days 91–120**: grain filling and maturity.
                3. **Yield output** — the panel on the right stays in view and updates instantly with
                   the predicted yield, yield per hectare and a 95% band.
                """
            )
    with g2:
        with st.container(border=True):
            st.markdown(
                """
                #### Batch CSV requirements
                Turn on **Batch CSV mode** in the sidebar, then upload a file whose headers match
                the training schema:
                - **Categoricals**: `Agriblock`, `Variety`, `Soil_Types`, `Nursery`, and `Wind_Direction_D*`
                - **Numerics**: `Hectares`, `Seedrate_in_Kg`, `Urea_40Days`, `Potassh_50Days`,
                  `Pest_60Day_in_ml`, `Trash_in_bundles`, and the interval climate columns
                  (`*Rain__in_mm`, `*AI_in_mm`, `Min_temp_*`, `Max_temp_*`, `Inst_Wind_Speed_*`,
                  `Relative_Humidity_*`).

                The full list is available inside the batch panel.
                """
            )


def page_about() -> None:
    """Project background carried over from ML_Paddy_pipeline(Regression).ipynb."""
    st.markdown(
        '<div class="app-hero"><h1>About this project</h1>'
        '<p>Agricultural Yield Intelligence System using Machine Learning</p></div>',
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2, gap="large")
    with a1:
        with st.container(border=True):
            st.markdown(
                """
                #### Problem statement
                Agricultural productivity, especially paddy cultivation, is highly influenced by
                many environmental and agronomic factors — rainfall, temperature, humidity, soil
                type, nursery conditions and wind. Predicting yield in advance is hard because
                these variables interact in complex ways.
                """
            )
    with a2:
        with st.container(border=True):
            st.markdown(
                """
                #### Goal
                Build a **machine learning model** that predicts actual paddy yield (in kilograms)
                from historical agronomic and climatic data — helping farmers, agricultural
                planners and policymakers make informed decisions on irrigation, fertilizer use
                and resource allocation.
                """
            )

    with st.container(border=True):
        section_title("🗂️", "Dataset", "Paddy cultivation field records from Tamil Nadu, India")
        st.markdown(
            f"""
            - **Records:** 2,789 raw rows → **2,338** after removing 451 exact duplicates.
            - **Columns:** 45 raw fields → **{len(TRAIN_COLS)} model features** after dropping the
              target and six highly-correlated / redundant columns identified in EDA.
            - **Target:** `Paddy_yield_in_Kg` — total paddy yield in kilograms (what we predict).
            """
        )
        d1, d2 = st.columns(2, gap="large")
        with d1:
            st.markdown(
                """
                **Field & agronomic features**
                | Column | Description |
                |---|---|
                | Hectares | Land area cultivated (hectares) |
                | Agriblock | Agricultural block / region |
                | Variety | Paddy variety (e.g. CO_43, ponmani) |
                | Soil_Types | Soil type (alluvial, clay, loamy…) |
                | Seedrate_in_Kg | Seed used per hectare |
                | Nursery | Nursery method (dry / wet) |
                | Urea_40Days | Urea applied at 40 days |
                | Potassh_50Days | Potash applied at 50 days |
                | Pest_60Day_in_ml | Pesticide at 60 days (ml) |
                | Trash_in_bundles | Crop residue in bundles |
                """
            )
        with d2:
            st.markdown(
                """
                **Weather features** (per 30-day stage: D1–D30, D31–D60, D61–D90, D91–D120)
                | Column | Description |
                |---|---|
                | *Rain__in_mm | Rainfall (mm) in the interval |
                | *AI_in_mm | Irrigation (mm) in the interval |
                | Min_temp / Max_temp | Min & max temperature |
                | Inst_Wind_Speed | Wind speed (knots) |
                | Wind_Direction | Wind direction |
                | Relative_Humidity | Relative humidity (%) |
                """
            )

    with st.container(border=True):
        section_title("🔬", "Method & key findings", "From exploratory analysis to the tuned pipeline")
        st.markdown(
            """
            - **Pipeline:** clean columns → drop duplicates → drop six multicollinear columns
              (correlation > 0.9, e.g. `LP_Mainfield`, `Nursery_area`, `DAP_20days`) →
              `train_test_split` (80/20) → IQR outlier clipping → `ColumnTransformer`
              (StandardScaler + OneHotEncoder) → `SelectKBest(f_regression)` → estimator.
            - **Models:** nine regressors (Linear, Ridge, Lasso, Decision Tree, Random Forest,
              Gradient Boosting, KNN, SVR, XGBoost) each tuned with `GridSearchCV` (cv=5, R²).
            - **Strongest predictor:** `Hectares` — yield rises almost linearly with area
              cultivated. Rainfall varies little across records, so it alone is a weak predictor.
            - **Soil & variety:** alluvial soil shows a slightly higher median yield; the three
              varieties have broadly similar yield distributions — inputs and weather matter more.
            - **Best model:** XGBoost (R² ≈ 0.99 on the holdout set). Full per-model diagnostics
              are on the **Model Info** page.
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — brand, navigation, secondary controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🌾</div>
            <div>
                <div class="brand-name">Rice Yield Predictor</div>
                <div class="brand-sub">Agricultural intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Predict Yield", "Model Info", "Guide", "About"],
        label_visibility="collapsed",
    )

    # ── Model picker (switch the active model at runtime) ──
    if model_loaded:
        st.markdown('<div class="nav-label">Model</div>', unsafe_allow_html=True)
        model_names = [d for d, _p, _k in AVAILABLE_MODELS]
        SELECTED_NAME = st.selectbox(
            "Active model", model_names, key="sel_model",
            help="Switch the model used for predictions, importance and residuals.",
        )
        _disp, _path, _key = next(m for m in AVAILABLE_MODELS if m[0] == SELECTED_NAME)
        model = load_model(_path)
        SELECTED_KEY = _key
        MODEL_PERF = perf_for(_key)
        CONFIDENCE_HALF_WIDTH = 2 * MODEL_PERF["rmse"]  # ~95% band
        st.markdown(
            f'<div class="brand-sub" style="padding:0 0.25rem;">'
            f'R² {MODEL_PERF["r2"]:.3f} · RMSE {MODEL_PERF["rmse"]:,.0f} Kg</div>',
            unsafe_allow_html=True,
        )
        if MODEL_LOAD_ERRORS:
            st.caption(f"{len(MODEL_LOAD_ERRORS)} model(s) unavailable — see Model Info.")

    batch_mode = False
    if page == "Predict Yield" and model_loaded:
        st.markdown('<div class="nav-label">Controls</div>', unsafe_allow_html=True)
        batch_mode = st.toggle("Batch CSV mode",
                               help="Switch between manual entry and CSV upload.")
        if st.button("Reset inputs", type="secondary"):
            reset_inputs()

# ══════════════════════════════════════════════════════════════════════════════
# Main content
# ══════════════════════════════════════════════════════════════════════════════
if not model_loaded:
    st.error("⚠️ Could not start the app. Ensure `data/paddydataset.csv` and at least one "
             "model in `models/` are present and loadable.")
    st.caption(f"Details: {load_error}")
    st.stop()

if page == "Predict Yield":
    page_predict(batch_mode)
elif page == "Model Info":
    page_model_info()
elif page == "Guide":
    page_guide()
else:
    page_about()

st.markdown(
    '<div style="margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #E3E6EA;'
    ' text-align: center; color: #5B6472; font-size: 0.8rem;">'
    'Rice Yield Prediction Engine · Agricultural Data Science Solutions</div>',
    unsafe_allow_html=True,
)
