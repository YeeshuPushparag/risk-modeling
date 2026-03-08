# bonds_train.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import os

# ===== PATHS CONFIGURATION =====
data_folder = os.path.join(os.path.dirname(__file__), 'data', 'final')
csv_file = os.path.join(data_folder, 'bonds.csv')
models_folder = os.path.join(os.path.dirname(__file__), 'models')


print("\n=== LOADING DATA ===")
df = pd.read_csv(csv_file)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["bond_id", "date"]).reset_index(drop=True)

# ---------------------------------------------
# FEATURE LIST (derived from your dataset)
# ---------------------------------------------
bond_features = [
    "credit_spread",
    "yield_to_maturity",
    "bond_price",
    "implied_hazard",
    "implied_pd_annual",
    "implied_pd_multi_year",
    "market_synthetic_score",
    "coupon_rate",
    "maturity_years",
    "benchmark_yield",
    "corporate_yield",
    # macro
    "gdp", "unrate", "cpi", "fedfunds"
]

bond_features = [c for c in bond_features if c in df.columns]

print("Using features:", bond_features)

# =====================================================
# MODEL 1 — 5-DAY CREDIT SPREAD FORECAST
# =====================================================
print("\n=== TRAINING MODEL 1: 5-DAY CREDIT SPREAD FORECAST ===")

df["target_spread_5d"] = df.groupby("bond_id")["credit_spread"].shift(-5)
ml1 = df[["date", "bond_id"] + bond_features + ["target_spread_5d"]].copy()
ml1 = ml1.replace([np.inf, -np.inf], np.nan).dropna()

X = ml1[bond_features].values
y = ml1["target_spread_5d"].values

split = int(len(ml1) * 0.80)
val = int(split * 0.90)

X_tr, y_tr = X[:val], y[:val]
X_val, y_val = X[val:split], y[val:split]
X_te, y_te = X[split:], y[split:]

dtr = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dte  = xgb.DMatrix(X_te, label=y_te)

params = {
    "objective": "reg:squarederror",
    "eta": 0.02,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "eval_metric": "rmse"
}

model_spread = xgb.train(
    params,
    dtr,
    num_boost_round=3000,
    evals=[(dtr,"train"), (dval,"val")],
    early_stopping_rounds=80,
    verbose_eval=200
)

pred_spread = model_spread.predict(dte)

mse = mean_squared_error(y_te, pred_spread)
r2  = r2_score(y_te, pred_spread)
ic  = spearmanr(pred_spread, y_te).correlation

print("\n--- MODEL 1 RESULTS ---")
print("MSE:", mse)
print("R2 :", r2)
print("Spearman IC:", ic)

model_spread.save_model(os.path.join(models_folder, "bond_model_spread5d_xgb.json"))
joblib.dump(bond_features, os.path.join(models_folder, "bond_features_spread5d.pkl"))
print("Saved spread model + features.")


# =====================================================
# MODEL 2 — 21-DAY PD FORECAST
# =====================================================
print("\n=== TRAINING MODEL 2: 21-DAY PD FORECAST ===")

df["target_pd_21d"] = df.groupby("bond_id")["implied_pd_annual"].shift(-21)

ml2 = df[["date", "bond_id"] + bond_features + ["target_pd_21d"]].copy()
ml2 = ml2.replace([np.inf, -np.inf], np.nan).dropna()

X = ml2[bond_features].values
y = ml2["target_pd_21d"].values

split = int(len(ml2) * 0.80)
val = int(split * 0.90)

X_tr, y_tr = X[:val], y[:val]
X_val, y_val = X[val:split], y[val:split]
X_te, y_te = X[split:], y[split:]

dtr = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dte  = xgb.DMatrix(X_te, label=y_te)

params_pd = {
    "objective": "reg:squarederror",
    "eta": 0.02,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "eval_metric": "rmse"
}

model_pd = xgb.train(
    params_pd,
    dtr,
    num_boost_round=4000,
    evals=[(dtr,"train"), (dval,"val")],
    early_stopping_rounds=100,
    verbose_eval=200
)

pred_pd = model_pd.predict(dte)

mse = mean_squared_error(y_te, pred_pd)
r2  = r2_score(y_te, pred_pd)
ic  = spearmanr(pred_pd, y_te).correlation

print("\n--- MODEL 2 RESULTS ---")
print("MSE:", mse)
print("R2 :", r2)
print("Spearman IC:", ic)

model_pd.save_model(os.path.join(models_folder, "bond_model_pd21d_xgb.json"))
joblib.dump(bond_features, os.path.join(models_folder, "bond_features_pd21d.pkl"))
print("Saved PD model + features.")

print("\n=== ALL BOND MODELS TRAINED SUCCESSFULLY ===")