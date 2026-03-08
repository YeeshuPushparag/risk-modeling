import pandas as pd
import numpy as np
import math
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import joblib
import os

# ===== PATHS CONFIGURATION =====
data_folder = os.path.join(os.path.dirname(__file__), 'data', 'final')
csv_file = os.path.join(data_folder, 'commodities.csv')
models_folder = os.path.join(os.path.dirname(__file__), 'models')


print("\n=== LOADING DATA ===")
df = pd.read_csv(csv_file)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
print("Total rows:", len(df))

# ============================
# 1) TARGET: 21-DAY FORWARD VOLATILITY
# ============================
def forward_realized_vol(series, window=21):
    f = series.shift(-1)             # future returns start next day
    rv = f.rolling(window).std()
    return rv * math.sqrt(252)       # annualize

df["target_vol21"] = df.groupby("ticker")["daily_return"].transform(
    lambda s: forward_realized_vol(s, 21)
)

df = df.dropna(subset=["target_vol21"]).reset_index(drop=True)
print("Rows after target creation:", len(df))

# ============================
# 2) FEATURES (13 commodity + 4 macro = 17)
# ============================
commodity_cols = [
    "daily_return", "log_return", "vol_20d",
    "sensitivity", "hedge_ratio", "exposure_amount",
    "commodity_pnl", "VaR_95", "VaR_99",
    "open", "high", "low", "close"
]

macro_cols = ["gdp", "unrate", "cpi", "fedfunds"]

feature_cols = [c for c in (commodity_cols + macro_cols) if c in df.columns]

print("Using {} features:".format(len(feature_cols)))
print(feature_cols)

# ============================
# 3) BUILD ML DATA (keep ticker & date)
# ============================
ml = df[["date", "ticker"] + feature_cols + ["target_vol21"]].copy()
ml = ml.replace([np.inf, -np.inf], np.nan)
ml = ml.dropna(subset=feature_cols + ["target_vol21"]).reset_index(drop=True)

print("Final training rows:", len(ml))

# ============================
# 4) TIME SPLIT (80/10/10 IB-style)
# ============================
X = ml[feature_cols].values
y = ml["target_vol21"].values

split = int(len(ml) * 0.80)
val = int(split * 0.90)

X_tr, y_tr = X[:val], y[:val]
X_val, y_val = X[val:split], y[val:split]
X_te, y_te = X[split:], y[split:]

print("Train:", X_tr.shape, "Val:", X_val.shape, "Test:", X_te.shape)

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_te,  label=y_te)

# ============================
# 5) MODEL PARAMS (optimized)
# ============================
params = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": 42
}

print("\n=== TRAINING COMMODITY VOL21 ===")
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=2500,
    evals=[(dtrain,"train"), (dval,"val")],
    early_stopping_rounds=80,
    verbose_eval=100
)

# ============================
# 6) PERFORMANCE
# ============================
pred = booster.predict(dtest)

mse = mean_squared_error(y_te, pred)
r2  = r2_score(y_te, pred)
ic  = spearmanr(pred, y_te).correlation

print("\n=== RESULTS ===")
print("MSE:", mse)
print("R2 :", r2)
print("Spearman IC:", ic)
print("Best iteration:", booster.best_iteration)

# Show sample predictions (no CSV)
sample = ml.iloc[split:split+5][["ticker","date","target_vol21"]].copy()
sample["pred_vol21"] = pred[:5]
print("\nSample predictions:")
print(sample)

# ============================
# 7) SAVE MODEL + FEATURES
# ============================
booster.save_model(os.path.join(models_folder, "commodities_model_vol21_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "commodities_features_vol21.pkl"))

print("\nSaved model + features.")
print("=== DONE ===")