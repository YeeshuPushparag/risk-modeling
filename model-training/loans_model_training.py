import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ===== PATHS CONFIGURATION =====
data_folder = os.path.join(os.path.dirname(__file__), 'data', 'final')
csv_file = os.path.join(data_folder, 'loans.csv')
models_folder = os.path.join(os.path.dirname(__file__), 'models')


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(csv_file)

# Ensure date format
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

cat_cols = ["ticker", "sector", "industry", "credit_rating", "rate_type", "counterparty"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Save encoders
joblib.dump(label_encoders, os.path.join(models_folder, "loans_label_encoders.pkl"))

# ==========================================
# TARGET — CREDIT SPREAD (the ONLY real target)
# ==========================================

target = "credit_spread"

# ==========================================
# FEATURES FOR CREDIT SPREAD MODEL
# ==========================================

features_cs = [
    # categorical (encoded)
    "ticker", "sector", "industry", "credit_rating", "rate_type", "counterparty",

    # loan economics
    "coupon_rate", "spread_rate", "notional_usd",
    "yield_to_maturity", "loan_age_months", "time_to_maturity_months",

    # FX linkage
    "fx_rate", "fx_volatility", "carry_daily",

    # market
    "close", "vol_20d",

    # macro
    "gdp", "unrate", "cpi", "fedfunds",

    # risk & exposures
    "exposure_pct_collateralized", "macro_stress_score",
    "volatility_index", "credit_spread_ratio",
    "profitability_ratio", "utilization_ratio",

    # cost, liquidity
    "funding_cost", "liquidity_score",
]

# Ensure all features exist
features_cs = [f for f in features_cs if f in df.columns]

# Filter dataset
ml = df[["date"] + features_cs + [target]].copy()
ml = ml.replace([np.inf, -np.inf], np.nan).dropna()

# ==========================================
# BUILD TRAIN / TEST SPLIT (TIME-SERIES)
# ==========================================

X = ml[features_cs].values
y = ml[target].values

split_idx = int(len(ml) * 0.80)
val_idx   = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te   = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_te, label=y_te)

# ==========================================
# MODEL PARAMETERS — same style as equity
# ==========================================

params = {
    "objective": "reg:squarederror",
    "eta": 0.01,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
}

# ==========================================
# TRAIN MODEL
# ==========================================

watchlist = [(dtrain, "train"), (dval, "val")]

print("\n=== TRAINING LOAN CREDIT SPREAD MODEL ===")

model_cs = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=200
)

# ==========================================
# EVALUATE ON TEST
# ==========================================

pred_cs = model_cs.predict(dtest)

mse_cs = mean_squared_error(y_te, pred_cs)
r2_cs  = r2_score(y_te, pred_cs)

print("\nCREDIT SPREAD MODEL - MSE:", mse_cs)
print("CREDIT SPREAD MODEL - R2 :", r2_cs)

# ==========================================
# SAVE MODEL + FEATURES
# ==========================================

model_cs.save_model(os.path.join(models_folder, "loans_model_creditspread_xgb.json"))
joblib.dump(features_cs, os.path.join(models_folder, "loans_features.pkl"))

print("\n=== LOAN CREDIT SPREAD MODEL SAVED ===")
print("Model saved: loans_model_creditspread_xgb.json")
print("Feature list saved: loans_features.pkl")
print("Encoders saved: loans_label_encoders.pkl")
print("=== DONE ===")