# models/train_fx_vol21.py
"""
Train FX 21-day realized volatility model (IB style)
"""

import math
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import os

# ===== PATHS CONFIGURATION =====
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'final')
csv_file = os.path.join(data_folder, 'fx.csv')
models_folder = os.path.join(os.path.dirname(__file__))



# -----------------------
# LOAD FX DATA
# -----------------------
df = pd.read_csv(csv_file, parse_dates=["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# -----------------------
# TARGET: 21-day FORWARD REALIZED VOLATILITY
# -----------------------
def forward_realized_vol(series, window=21):
    fwd = series.shift(-1)                    # start from tomorrow
    rv  = fwd.rolling(window).std()           # 21-day realized vol
    return rv * math.sqrt(252)                # annualize

df["target_fx_vol21"] = df.groupby("ticker")["fx_return"].transform(
    lambda s: forward_realized_vol(s, 21)
)

# -----------------------
# FINAL FEATURES
# -----------------------
feature_cols = [
    "fx_return", "fx_volatility_20d", "fx_volatility_30d", "fx_volatility",
    "interest_diff", "carry_daily", "return_carry_adj",
    "position_size", "hedge_ratio", "exposure_amount",
    "fx_pnl", "carry_pnl", "total_pnl", "expected_pnl",
    "VaR_95", "VaR_99", "value_at_risk",
    "volume", "sharpe_like_ratio",
    # macro
    "gdp", "unrate", "cpi", "fedfunds",
    # misc
    "foreign_revenue_ratio"
]

feature_cols = [c for c in feature_cols if c in df.columns]

# -----------------------
# CLEAN ML DATASET (KEEP TICKER + DATE)
# -----------------------
ml = df[["ticker", "date"] + feature_cols + ["target_fx_vol21"]].copy()
ml = ml.replace([np.inf, -np.inf], np.nan)
ml = ml.dropna(subset=feature_cols + ["target_fx_vol21"])

print(f"Rows: {len(ml)}  | Features used: {len(feature_cols)}")

# -----------------------
# TIME-SERIES SPLIT
# -----------------------
split_idx = int(len(ml) * 0.80)
val_idx   = split_idx + int((len(ml) - split_idx) * 0.50)

X = ml[feature_cols].values
y = ml["target_fx_vol21"].values

X_tr, y_tr = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:val_idx], y[split_idx:val_idx]
X_te, y_te = X[val_idx:], y[val_idx:]

print("Train:", X_tr.shape, "Val:", X_val.shape, "Test:", X_te.shape)

# -----------------------
# XGBOOST TRAINING
# -----------------------
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "seed": 42,
    "tree_method": "hist"
}

print("\n=== TRAINING FX VOL21 MODEL ===")
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=2500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=200
)

# -----------------------
# EVALUATION
# -----------------------
pred = booster.predict(dtest)

mse = mean_squared_error(y_te, pred)
r2  = r2_score(y_te, pred)
ic  = spearmanr(pred, y_te).correlation

print("\n=== FX VOL21 RESULTS ===")
print("MSE:", mse)
print("R2 :", r2)
print("Spearman IC:", ic)
print("Best iteration:", booster.best_iteration)

# -----------------------
# SAVE MODEL + FEATURE LIST
# -----------------------
booster.save_model(os.path.join(models_folder, "fx_model_vol21_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "fx_features_vol21.pkl"))

print("\nSaved model: fx_model_vol21_xgb.json")
print("Saved features:", "fx_features_vol21.pkl")
print("=== DONE ===")