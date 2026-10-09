# Rice Yield Predictor

We built a machine-learning pipeline that predicts the total paddy (rice) yield of a cultivated field, in kilograms, from its agronomic and climatic conditions. We deployed the best model as an interactive Streamlit application for single and batch predictions.

This was completed as a Data Mining Lab academic project at Prasad V. Potluri Siddhartha Institute of Technology (2026–27).

## Highlights

- We trained nine regression models on one identical, leakage-free pipeline and tuned each with 5-fold GridSearchCV.
- Our best model is an **XGBoost Regressor**, with a test R² of 0.9904, an RMSE of about 905 Kg and an MAE of about 656 Kg.
- We found that hectares is the dominant predictor, while rainfall and soil type are weak individually.
- We built a Streamlit app with single-field prediction, batch CSV prediction, model comparison, and a guide.

## Problem

Paddy productivity depends on many interacting factors: rainfall, temperature, humidity, wind, soil, seed rate, nursery method, and the timing of fertilizer and pesticide application. Rule-of-thumb forecasting struggles with these non-linear interactions. We wanted to learn these relationships from historical field records so that yield can be estimated before harvest, supporting irrigation planning, fertilizer use, labour allocation and supply planning.

## Dataset

Our data consists of real paddy-cultivation field records from six agricultural blocks in Tamil Nadu, India: Cuddalore, Kurinjipadi, Panruti, Kallakurichi, Sankarapuram and Chinnasalem. Each row is one cultivated field.

- **Raw size:** 2,789 rows × 45 fields
- **After de-duplication:** 2,338 rows (we removed 451 exact duplicates)
- **Model features:** 38 (after removing the target and six multicollinear columns)
- **Target:** `Paddy_yield_in_Kg`

**Feature groups**

| Group | Examples |
|---|---|
| Land and crop | `Hectares`, `Agriblock`, `Variety` (CO_43, ponmani, delux ponni), `Soil_Types` (alluvial, clay), `Nursery` (dry, wet) |
| Inputs | `Seedrate_in_Kg`, `Urea_40Days`, `Potassh_50Days`, `Pest_60Day_in_ml`, `Trash_in_bundles` |
| Weather (per 30-day stage, D1–D120) | Rainfall and irrigation (mm), min and max temperature, instantaneous wind speed (knots), wind direction, relative humidity |

We standardised the column names during cleaning, so headers such as `30DRain__in_mm` and `Inst_Wind_Speed_D1_D30_in_Knots` are used throughout the code.

## Pipeline

We applied the same pipeline to every model:

1. **Cleaning:** we standardised column names, removed 451 duplicate rows, and dropped six highly correlated columns (`LP_Mainfield_in_Tonnes`, `LP_nurseryarea_in_Tonnes`, `Nursery_area__Cents`, `DAP_20days`, `Weed28D_thiobencarb`, `Micronutrients_70Days`).
2. **Split:** an 80:20 train–test split with `random_state = 42` (1,870 train / 468 test).
3. **Outliers:** IQR clipping, with limits computed on the training set only and applied to the test set.
4. **Transform:** a `ColumnTransformer` with `StandardScaler` on numeric columns and `OneHotEncoder` on categorical columns.
5. **Feature selection:** `SelectKBest` with `f_regression`. The number of features *k* (10, 15 or 20) is tuned by the grid search.
6. **Model:** one of nine regressors.
7. **Tuning:** `GridSearchCV` with 5-fold cross-validation, scored on R².

Every step that learns from the data is fitted on the training set only, so no information leaks from the test set.

### Models compared

Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN, SVR and XGBoost.

## Results

Test-set performance on the 468-row hold-out set, sorted by R²:

| Model | Test R² | RMSE (Kg) | MAE (Kg) |
|---|---|---|---|
| **XGBoost** | **0.9904** | **905** | **656** |
| Random Forest | 0.9903 | 914 | 659 |
| Gradient Boosting | 0.9902 | 915 | 659 |
| Decision Tree | 0.9902 | 915 | 660 |
| KNN | 0.9890 | 970 | 701 |
| Lasso | 0.9878 | 1,022 | 763 |
| Linear Regression | 0.9878 | 1,022 | 763 |
| Ridge | 0.9878 | 1,022 | 763 |
| SVR | 0.9864 | 1,080 | 775 |

Our selected XGBoost configuration is `n_estimators = 100`, `max_depth = 3`, `learning_rate = 0.1`, with k = 20 features. The differences among the top tree-based models are small, but XGBoost was consistently best on every metric.

**Key findings**

- Hectares is the strongest predictor, and our feature-importance plot confirms it.
- Rainfall varies little across our records (about 18–20 mm) and is a weak individual predictor.
- Soil type and variety have only modest effects on yield.
- The target is right-skewed, with a median near 25,000 Kg.

## Application

We saved the trained pipeline as `paddy_yield_model.pkl`, which the Streamlit app loads once. The app has four pages:

- **Predict Yield:** enter a farm profile and four stages of weather to get a predicted yield, yield per hectare, an approximate 95% band (±2 × RMSE), and a HIGH / MEDIUM / LOW category with a short recommendation.
- **Batch Predictions:** upload a CSV with one row per field. The app checks for all 38 required columns, predicts each row with lower and upper bounds, shows summary statistics, and returns a downloadable CSV.
- **Model Info:** switch among the nine saved pipelines, and inspect metrics, a benchmark table, feature importance and residuals.
- **Guide and About:** instructions for single and batch use, the batch CSV format, and a summary of the problem and findings.

## Installation

Requires Python 3.9 or later.

```bash
git clone https://github.com/mariognan13-lab/Rice-yield-predictor.git
cd Rice-yield-predictor
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the main dependencies directly:

```bash
pip install scikit-learn==1.9.1 xgboost==3.4.1 streamlit pandas numpy matplotlib seaborn altair
```

## Usage

Run the application:

```bash
streamlit run app.py
```

For batch predictions, prepare a CSV with the 38 feature columns listed in the Guide page, then upload it on the Batch Predictions page.

## Limitations and Future Work

- We trained on one region's records, so the model may not generalise to other regions, seasons or varieties without retraining.
- Our predictions use only weather and agronomic inputs. We have not yet included satellite indices (such as NDVI), soil-nutrient tests or daily weather.
- Our confidence bands are approximations based on RMSE, not true prediction intervals.
- Next, we plan to explore sequence models (such as LSTMs) for weather dynamics, quantile or probabilistic models for proper intervals, and multi-season and multi-region support.

## Team

- U.S.V. Rishi Raj (24501A05O4): [GitHub](https://github.com/rishirajunguturi-bit/Rice-Yeild-Predictor)
- S. Mario Gnan (25505A0520): [GitHub](https://github.com/mariognan13-lab/Rice-yield-predictor)
- Shaik Moulana (24501A05K8): [GitHub](https://github.com/shaikmubasheer006/Rice-yield-predictor)
- S. Pavani (24501A05M8): [GitHub](https://github.com/PavaniSykam/Rice-yield-predictor)

## References

- scikit-learn: https://scikit-learn.org/stable/
- XGBoost: https://xgboost.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
