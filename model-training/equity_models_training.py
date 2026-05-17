import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import joblib
import math
import os

# ===== PATHS CONFIGURATION =====
data_folder = os.path.join(os.path.dirname(__file__), 'data', 'final')
csv_file = os.path.join(data_folder, 'equity.csv')
models_folder = os.path.join(os.path.dirname(__file__), 'models')

# Create models folder if it doesn't exist
os.makedirs(models_folder, exist_ok=True)

# ===== DATA LOADING & PREPARATION =====
df = pd.read_csv(csv_file)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['asset_manager', 'ticker', 'date']).reset_index(drop=True)

# Create target returns
df['target_ret_1d'] = df.groupby('ticker')['daily_return'].shift(-1)
df['target_ret_5d'] = df.groupby('ticker')['daily_return'].shift(-5)
df['target_ret_21d'] = df.groupby('ticker')['daily_return'].shift(-21)
df = df.dropna(subset=['target_ret_1d', 'target_ret_5d', 'target_ret_21d'])

# Feature selection
feature_cols = [
    'daily_return', 'log_return', 'high_low_spread', 'close_open_diff',
    'vol_5d', 'vol_20d', 'volatility_21d', 'ma_5', 'ma_20', 'momentum_5d', 'momentum_20d',
    'avg_volume_10d', 'vol_change', 'beta', 'downside_risk', 'sharpe_ratio', 'sortino_ratio',
    'daily_sigma', 'daily_mu', 'daily_VaR_95', 'daily_VaR_99', 'daily_CVaR_95', 'daily_CVaR_99',
    'portfolio_weight', 'sector_exposure', 'industry_exposure', 'beta_weighted', 'sector_beta_weighted',
    'gdp', 'unrate', 'cpi', 'fedfunds', 'totalDebt', 'shortTermDebt', 'longTermDebt', 'totalAssets',
    'ebitda', 'revenue', 'netIncome', 'debt_to_assets', 'debt_to_ebitda', 'ebitda_margin', 'marketCap',
    'Amihud_illiquidity', 'turnover_ratio', 'liquidity_risk_score', 'top_5_exposure', 'HHI_sector', 'diversification_score'
]
feature_cols = [c for c in feature_cols if c in df.columns]

# ===== 1-DAY RETURN MODEL =====
print("=== TRAINING 1-DAY RETURN MODEL ===")
target_col = "target_ret_1d"
ml1 = df[feature_cols + [target_col]].copy()
ml1 = ml1.replace([np.inf, -np.inf], np.nan).dropna()

X = ml1[feature_cols].values
y = ml1[target_col].values

split_idx = int(len(ml1) * 0.80)
X_tr, X_te = X[:split_idx], X[split_idx:]
y_tr, y_te = y[:split_idx], y[split_idx:]

params = {
    "n_estimators": 2000, "max_depth": 4, "learning_rate": 0.06,
    "subsample": 0.9, "colsample_bytree": 0.8, "objective": "reg:squarederror",
    "tree_method": "hist", "random_state": 42
}

model_1d = xgb.XGBRegressor(**params)
model_1d.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=200)
pred_1d = model_1d.predict(X_te)

mse_1d = mean_squared_error(y_te, pred_1d)
r2_1d = r2_score(y_te, pred_1d)
print("1-DAY MODEL - MSE:", mse_1d, "R2:", r2_1d)

# Post-training analysis
preds_1d = pd.DataFrame({"true_return": y_te, "pred_return": pred_1d})
ic_1d, _ = spearmanr(preds_1d["true_return"], preds_1d["pred_return"])
print("Spearman IC (1-day):", ic_1d)

preds_1d["rank"] = preds_1d["pred_return"].rank(pct=True)
top = preds_1d[preds_1d["rank"] >= 0.80]["true_return"].mean()
bottom = preds_1d[preds_1d["rank"] <= 0.20]["true_return"].mean()
ls_return = top - bottom
print("Top-Bottom Daily:", ls_return)

model_1d.get_booster().save_model(os.path.join(models_folder, "equity_model_1d_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "equity_features_1d.pkl"))

# ===== 5-DAY RETURN MODEL =====
print("\n=== TRAINING 5-DAY RETURN MODEL ===")
target_col = 'target_ret_5d'
ml = df[feature_cols + [target_col]].copy()
ml = ml.replace([np.inf, -np.inf], np.nan).dropna()

X = ml[feature_cols].values
y = ml[target_col].values
split_idx = int(len(ml) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx] 
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror", "eta": 0.03, "max_depth": 4,
    "subsample": 0.9, "colsample_bytree": 0.8, "eval_metric": "rmse"
}

watchlist = [(dtrain, "train"), (dval, "val")]
booster_5d = xgb.train(params, dtrain, num_boost_round=2000, evals=watchlist,
                      early_stopping_rounds=50, verbose_eval=50)
pred_5d = booster_5d.predict(dtest)

print("5-DAY MODEL - MSE:", mean_squared_error(y_te, pred_5d), "R2:", r2_score(y_te, pred_5d))

# Save predictions with full analysis
df_test = df.iloc[split_idx: split_idx + len(pred_5d)].copy().reset_index(drop=True)
df_test['pred_ret_5d'] = pred_5d
ic_5d = spearmanr(df_test['pred_ret_5d'], df_test['target_ret_5d']).correlation
print("Spearman IC (5-day):", ic_5d)

df_test['pred_rank'] = df_test.groupby('date')['pred_ret_5d'].rank(pct=True)
longs = df_test[df_test['pred_rank'] >= 0.9].groupby('date')['target_ret_5d'].mean()
shorts = df_test[df_test['pred_rank'] <= 0.1].groupby('date')['target_ret_5d'].mean()
cs_return = (longs - shorts).dropna()
print("Top-bottom mean daily:", cs_return.mean())

booster_5d.save_model(os.path.join(models_folder, "equity_model_5d_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "equity_features_5d.pkl"))

# ===== 21-DAY RETURN MODEL =====
print("\n=== TRAINING 21-DAY RETURN MODEL ===")
target_col = "target_ret_21d"
ml21 = df[["date"] + feature_cols + [target_col]].copy()
ml21 = ml21.replace([np.inf, -np.inf], np.nan).dropna()

X = ml21[feature_cols].values
y = ml21[target_col].values
split_idx = int(len(ml21) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror", "eta": 0.01, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse"
}

watchlist = [(dtrain, "train"), (dval, "val")]
booster21 = xgb.train(params, dtrain, num_boost_round=5000, evals=watchlist,
                     early_stopping_rounds=100, verbose_eval=100)
pred21 = booster21.predict(dtest)

print("21-DAY MODEL - MSE:", mean_squared_error(y_te, pred21), "R2:", r2_score(y_te, pred21))

# Rebuild test dataframe with date for analysis
df_test21 = ml21.iloc[split_idx:].copy().reset_index(drop=True)
df_test21["pred_ret_21d"] = pred21
ic_21 = spearmanr(df_test21["pred_ret_21d"], df_test21["target_ret_21d"]).correlation
print("Spearman IC (21-day):", ic_21)

df_test21["pred_rank"] = df_test21["pred_ret_21d"].rank(pct=True)
top = df_test21[df_test21["pred_rank"] >= 0.80].groupby("date")["target_ret_21d"].mean()
bottom = df_test21[df_test21["pred_rank"] <= 0.20].groupby("date")["target_ret_21d"].mean()
cs_return21 = (top - bottom).dropna()
print("Daily long-short:", cs_return21.mean())

booster21.save_model(os.path.join(models_folder, "equity_model_21d_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "equity_features_21d.pkl"))

# ===== VOLATILITY FORECAST MODEL =====
print("\n=== TRAINING VOLATILITY FORECAST MODEL ===")
def forward_realized_vol(series, window=21):
    f = series.shift(-1)
    rv = f.rolling(window=window).std()
    return rv * math.sqrt(252)

df["target_vol21"] = df.groupby("ticker")["daily_return"].transform(lambda s: forward_realized_vol(s, 21))

feature_cols_vol = [c for c in feature_cols if c in df.columns]
ml_vol = df[feature_cols_vol + ["target_vol21", "ticker", "date"]].copy()
ml_vol = ml_vol.replace([np.inf, -np.inf], np.nan)
ml_vol = ml_vol.dropna(subset=["target_vol21"]).dropna(subset=feature_cols_vol)

X = ml_vol[feature_cols_vol].values
y = ml_vol["target_vol21"].values
split_idx = int(len(ml_vol) * 0.80)

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

params = {
    "objective": "reg:squarederror", "eta": 0.03, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "rmse"
}

model_vol = xgb.train(params, dtrain, num_boost_round=1500, evals=[(dtrain, "train")], verbose_eval=200)
pred_vol = model_vol.predict(dtest)

print("VOLATILITY MODEL - MSE:", mean_squared_error(y_test, pred_vol), "R2:", r2_score(y_test, pred_vol))

model_vol.save_model(os.path.join(models_folder, "equity_model_vol21_xgb.json"))
joblib.dump(feature_cols_vol, os.path.join(models_folder, "equity_features_vol21.pkl"))

# ===== DOWNSIDE RISK MODEL =====
print("\n=== TRAINING DOWNSIDE RISK MODEL ===")
def realized_downside(series, window):
    s = series.shift(-1)
    neg = s[s < 0]
    roll = neg.rolling(window=window).std()
    return roll

df["target_down21"] = df.groupby("ticker")["daily_return"].transform(lambda s: realized_downside(s, 21))
df = df.dropna(subset=["target_down21"])

features_down = feature_cols.copy()
ml_down = df[features_down + ["target_down21"]].replace([np.inf, -np.inf], np.nan).dropna()

X = ml_down[features_down].values
y = ml_down["target_down21"].values

split_idx = int(len(ml_down) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror", "eta": 0.01, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse"
}

booster_down = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dtrain, "train"), (dval, "val")],
                        early_stopping_rounds=120, verbose_eval=150)
pred_down = booster_down.predict(dtest)

print("DOWNSIDE RISK MODEL - MSE:", mean_squared_error(y_te, pred_down), "R2:", r2_score(y_te, pred_down))
booster_down.save_model(os.path.join(models_folder, "equity_model_down21_xgb.json"))
joblib.dump(features_down, os.path.join(models_folder, "equity_features_down21.pkl"))

# ===== VaR MODEL =====
print("\n=== TRAINING VaR MODEL ===")
Z = 1.65
df["target_var21"] = -(df["daily_mu"] + Z * df["volatility_21d"])

ml_var = df[feature_cols + ["target_var21"]].copy()
ml_var = ml_var.replace([np.inf, -np.inf], np.nan).dropna()

X = ml_var[feature_cols].values
y = ml_var["target_var21"].values

split_idx = int(len(ml_var) * 0.80)
val_idx = int(split_idx * 0.9)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror", "eta": 0.01, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse"
}

booster_var = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dtrain,"train"), (dval,"val")],
                       early_stopping_rounds=100, verbose_eval=200)
pred_var = booster_var.predict(dtest)

print("VaR MODEL - MSE:", mean_squared_error(y_te, pred_var), "R2:", r2_score(y_te, pred_var))
booster_var.save_model(os.path.join(models_folder, "equity_model_var21_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "equity_features_var21.pkl"))

# ===== FACTOR RETURN MODEL =====
print("\n=== TRAINING FACTOR RETURN MODEL ===")
df["fwd_21d"] = df.groupby("ticker")["daily_return"].shift(-21)
mkt21 = df.groupby("date")["daily_return"].mean().shift(-21)

df = df.merge(mkt21.rename("mkt_fwd_21d"), left_on="date", right_index=True, how="left")
df["target_factor_21d"] = df["fwd_21d"] - (df["beta"] * df["mkt_fwd_21d"])

df_factor = df[feature_cols + ["target_factor_21d"]].copy()
df_factor = df_factor.replace([np.inf, -np.inf], np.nan).dropna()

X = df_factor[feature_cols].values
y = df_factor["target_factor_21d"].values

split = int(len(df_factor) * 0.80)
val = int(split * 0.90)

X_tr, y_tr = X[:val], y[:val]
X_val, y_val = X[val:split], y[val:split]
X_te, y_te = X[split:], y[split:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror", "eta": 0.01, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse"
}

model_factor = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dtrain, "train"), (dval, "val")],
                        early_stopping_rounds=100, verbose_eval=200)
pred_factor = model_factor.predict(dtest)

print("FACTOR RETURN MODEL - MSE:", mean_squared_error(y_te, pred_factor), "R2:", r2_score(y_te, pred_factor))

ic_factor = spearmanr(pred_factor, y_te).correlation
print("Spearman IC (Factor):", ic_factor)

model_factor.save_model(os.path.join(models_folder, "equity_model_factor21_xgb.json"))
joblib.dump(feature_cols, os.path.join(models_folder, "equity_features_factor21.pkl"))

# ===== SECTOR ROTATION MODEL =====
print("\n=== TRAINING SECTOR ROTATION MODEL ===")
df_sec = df.copy()
df_sec["date"] = pd.to_datetime(df_sec["date"])
df_sec = df_sec.sort_values(["ticker", "date"])

df_sec["sector_ret_21d"] = df_sec.groupby("sector")["daily_return"].shift(-21)
df_sec = df_sec.dropna(subset=["sector_ret_21d"]).reset_index(drop=True)

df_sec["sector_avg_return"] = df_sec.groupby("sector")["daily_return"].transform("mean")
df_sec["sector_mom20"] = df_sec.groupby("sector")["sector_avg_return"].transform(lambda s: s.pct_change(20))
df_sec["sector_vol21"] = df_sec.groupby("sector")["sector_avg_return"].transform(lambda s: s.rolling(21).std())
df_sec["sector_turnover"] = df_sec.groupby("sector")["turnover_ratio"].transform("mean")

df_sec["gdp_trend"] = df_sec["gdp"].diff(63)
df_sec["cpi_trend"] = df_sec["cpi"].diff(63)
df_sec["unrate_trend"] = df_sec["unrate"].diff(63)
df_sec["fedfunds_trend"] = df_sec["fedfunds"].diff(63)
df_sec = df_sec.fillna(0)

sector_features = [
    "daily_return", "momentum_20d", "volatility_21d", "vol_20d", "vol_change",
    "avg_volume_10d", "turnover_ratio", "Amihud_illiquidity", "sharpe_ratio", 
    "sortino_ratio", "downside_risk", "beta_weighted", "sector_beta_weighted",
    "sector_exposure", "HHI_sector", "diversification_score", "sector_avg_return",
    "sector_mom20", "sector_vol21", "sector_turnover", "gdp", "unrate", "cpi", 
    "fedfunds", "gdp_trend", "cpi_trend", "unrate_trend", "fedfunds_trend"
]
sector_features = [c for c in sector_features if c in df_sec.columns]

X = df_sec[sector_features].values
y = df_sec["sector_ret_21d"].values

split_idx = int(len(df_sec) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params = {
    "objective": "reg:squarederror", "eta": 0.01, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse",
    "tree_method": "hist"
}

booster_sec = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dtrain, "train"), (dval, "val")],
                       early_stopping_rounds=200, verbose_eval=200)
pred_sec = booster_sec.predict(dtest)

print("SECTOR ROTATION MODEL - MSE:", mean_squared_error(y_te, pred_sec), "R2:", r2_score(y_te, pred_sec))

ic_sec = spearmanr(pred_sec, y_te).correlation
print("Spearman IC (Sector):", ic_sec)

booster_sec.save_model(os.path.join(models_folder, "equity_model_sector_rotation_xgb.json"))
joblib.dump(sector_features, os.path.join(models_folder, "equity_features_sector_rotation.pkl"))

# ===== MACRO REGIME MODEL =====
print("\n=== TRAINING MACRO REGIME MODEL ===")
macro_features = [
    "gdp", "cpi", "unrate", "fedfunds", "volatility_21d", "daily_sigma",
    "downside_risk", "daily_VaR_95", "daily_CVaR_95"
]

df_macro = df[macro_features].replace([np.inf, -np.inf], np.nan).dropna().copy()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_macro["regime"] = kmeans.fit_predict(df_macro[macro_features])
joblib.dump(kmeans, os.path.join(models_folder, "macro_kmeans.pkl"))

X = df_macro[macro_features].values
y = df_macro["regime"].values

split_idx = int(len(df_macro) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model_regime = xgb.XGBClassifier(
    n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
    num_class=4, eval_metric="mlogloss"
)

model_regime.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
preds_regime = model_regime.predict(X_test)

print("MACRO REGIME MODEL - Accuracy:", accuracy_score(y_test, preds_regime))
model_regime.save_model(os.path.join(models_folder, "equity_model_macro_regime_xgb.json"))
joblib.dump(macro_features, os.path.join(models_folder, "equity_features_macro_regime.pkl"))

# ===== CREATE PORTFOLIO TARGETS =====
print("\n=== CREATING PORTFOLIO TARGETS ===")
for N in [1, 5, 21]:
    df[f"fwd_ret_{N}d"] = df.groupby(["asset_manager","ticker"])["close"].shift(-N) / df["close"] - 1
    df[f"port_ret_contrib_{N}d"] = df["portfolio_weight"] * df[f"fwd_ret_{N}d"]

# ===== CREATE HISTORICAL FEATURES (NO LEAKAGE) =====
print("\n=== CREATING HISTORICAL FEATURES ===")
for N in [1, 5, 21]:
    df[f"lag_ret_{N}d"] = df.groupby(["asset_manager", "ticker"])["close"].pct_change(N)
    df[f"lag_port_ret_contrib_{N}d"] = df["portfolio_weight"] * df[f"lag_ret_{N}d"]

# ===== PORTFOLIO VaR 1D MODEL =====
print("\n=== TRAINING PORTFOLIO VaR 1D MODEL ===")
df['target_portfolio_var_1d'] = df.groupby('ticker')['daily_portfolio_VaR_95'].shift(-1)

portfolio_var_features = [
    'daily_portfolio_ex_ante_volatility', 'daily_portfolio_VaR_95', 'daily_portfolio_VaR_99',
    'daily_portfolio_CVaR_95', 'daily_portfolio_CVaR_99', 'daily_portfolio_mu',
    'portfolio_weight', 'sector_exposure', 'industry_exposure',
    'top_5_exposure', 'HHI_sector', 'diversification_score',
    'volatility_21d', 'vol_5d', 'vol_20d', 'daily_sigma', 'daily_mu',
    'downside_risk', 'sharpe_ratio', 'sortino_ratio',
    'turnover_ratio', 'Amihud_illiquidity', 'liquidity_risk_score',
    'gdp', 'unrate', 'cpi', 'fedfunds'
]
portfolio_var_features = [f for f in portfolio_var_features if f in df.columns]

target_var_1d = 'target_portfolio_var_1d'
ml_var_1d = df[portfolio_var_features + [target_var_1d]].copy()
ml_var_1d = ml_var_1d.replace([np.inf, -np.inf], np.nan).dropna()

X = ml_var_1d[portfolio_var_features].values
y = ml_var_1d[target_var_1d].values

split_idx = int(len(ml_var_1d) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params_var_1d = {
    "objective": "reg:squarederror", "eta": 0.03, "max_depth": 4,
    "subsample": 0.9, "colsample_bytree": 0.8, "eval_metric": "rmse"
}

watchlist = [(dtrain, "train"), (dval, "val")]
model_var_1d = xgb.train(params_var_1d, dtrain, num_boost_round=2500, evals=watchlist,
                        early_stopping_rounds=80, verbose_eval=100)
pred_var_1d = model_var_1d.predict(dtest)

print("PORTFOLIO VaR 1D - MSE:", mean_squared_error(y_te, pred_var_1d), "R2:", r2_score(y_te, pred_var_1d))
model_var_1d.save_model(os.path.join(models_folder, "equity_model_portfolio_var_1d_xgb.json"))
joblib.dump(portfolio_var_features, os.path.join(models_folder, "equity_features_portfolio_var_1d.pkl"))

# ===== PORTFOLIO VaR 5D MODEL =====
print("\n=== TRAINING PORTFOLIO VaR 5D MODEL ===")
df['target_portfolio_var_5d'] = df.groupby('ticker')['daily_portfolio_VaR_95'].shift(-5)

target_var_5d = 'target_portfolio_var_5d'
ml_var_5d = df[portfolio_var_features + [target_var_5d]].copy()
ml_var_5d = ml_var_5d.replace([np.inf, -np.inf], np.nan).dropna()

X = ml_var_5d[portfolio_var_features].values
y = ml_var_5d[target_var_5d].values

split_idx = int(len(ml_var_5d) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params_var_5d = {
    "objective": "reg:squarederror", "eta": 0.02, "max_depth": 4,
    "subsample": 0.85, "colsample_bytree": 0.75, "eval_metric": "rmse"
}

model_var_5d = xgb.train(params_var_5d, dtrain, num_boost_round=3000, evals=watchlist,
                        early_stopping_rounds=100, verbose_eval=100)
pred_var_5d = model_var_5d.predict(dtest)

print("PORTFOLIO VaR 5D - MSE:", mean_squared_error(y_te, pred_var_5d), "R2:", r2_score(y_te, pred_var_5d))
model_var_5d.save_model(os.path.join(models_folder, "equity_model_portfolio_var_5d_xgb.json"))
joblib.dump(portfolio_var_features, os.path.join(models_folder, "equity_features_portfolio_var_5d.pkl"))

# ===== PORTFOLIO VaR 21D MODEL =====
print("\n=== TRAINING PORTFOLIO VaR 21D MODEL ===")
df['target_portfolio_var_21d'] = df.groupby('ticker')['daily_portfolio_VaR_95'].shift(-21)

target_var_21d = 'target_portfolio_var_21d'
ml_var_21d = df[portfolio_var_features + [target_var_21d]].copy()
ml_var_21d = ml_var_21d.replace([np.inf, -np.inf], np.nan).dropna()

X = ml_var_21d[portfolio_var_features].values
y = ml_var_21d[target_var_21d].values

split_idx = int(len(ml_var_21d) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params_var_21d = {
    "objective": "reg:squarederror", "eta": 0.01, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse"
}

model_var_21d = xgb.train(params_var_21d, dtrain, num_boost_round=5000, evals=watchlist,
                         early_stopping_rounds=100, verbose_eval=200)
pred_var_21d = model_var_21d.predict(dtest)

print("PORTFOLIO VaR 21D - MSE:", mean_squared_error(y_te, pred_var_21d), "R2:", r2_score(y_te, pred_var_21d))
model_var_21d.save_model(os.path.join(models_folder, "equity_model_portfolio_var_21d_xgb.json"))
joblib.dump(portfolio_var_features, os.path.join(models_folder, "equity_features_portfolio_var_21d.pkl"))

# ===== PORTFOLIO 1D RETURN MODEL =====
print("\n=== TRAINING PORTFOLIO 1D RETURN MODEL ===")

portfolio_features_1d = (
    feature_cols +
    [
        "lag_ret_5d",
        "lag_port_ret_contrib_5d"
    ]
)

portfolio_features_1d = [
    f for f in portfolio_features_1d
    if f in df.columns
]

target_1d = "port_ret_contrib_1d"

ml_1d = df[
    portfolio_features_1d + [target_1d]
].copy()

ml_1d = (
    ml_1d
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

X = ml_1d[portfolio_features_1d].values
y = ml_1d[target_1d].values

split_idx = int(len(ml_1d) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

params_port = {
    "objective": "reg:squarederror",
    "eta": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "eval_metric": "rmse",
    "tree_method": "hist"
}

watchlist = [
    (dtrain, "train"),
    (dval, "val")
]

model_1d_port = xgb.train(
    params_port,
    dtrain,
    num_boost_round=4000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=200
)

pred_1d_port = model_1d_port.predict(dtest)

print(
    "PORTFOLIO 1D - MSE:",
    mean_squared_error(y_te, pred_1d_port),
    "R2:",
    r2_score(y_te, pred_1d_port)
)

model_1d_port.save_model(
    os.path.join(models_folder, "equity_model_portfolio1d_xgb.json")
)

joblib.dump(
    portfolio_features_1d,
    os.path.join(models_folder, "equity_features_portfolio1d.pkl")
)

# ===== PORTFOLIO 5D RETURN MODEL =====
print("\n=== TRAINING PORTFOLIO 5D RETURN MODEL ===")

portfolio_features_5d = (
    feature_cols +
    [
        "lag_ret_1d",
        "lag_port_ret_contrib_1d"
    ]
)

portfolio_features_5d = [
    f for f in portfolio_features_5d
    if f in df.columns
]

target_5d = "port_ret_contrib_5d"

ml_5d = df[
    portfolio_features_5d + [target_5d]
].copy()

ml_5d = (
    ml_5d
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

X = ml_5d[portfolio_features_5d].values
y = ml_5d[target_5d].values

split_idx = int(len(ml_5d) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

watchlist = [
    (dtrain, "train"),
    (dval, "val")
]

model_5d_port = xgb.train(
    params_port,
    dtrain,
    num_boost_round=4000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=200
)

pred_5d_port = model_5d_port.predict(dtest)

print(
    "PORTFOLIO 5D - MSE:",
    mean_squared_error(y_te, pred_5d_port),
    "R2:",
    r2_score(y_te, pred_5d_port)
)

model_5d_port.save_model(
    os.path.join(models_folder, "equity_model_portfolio5d_xgb.json")
)

joblib.dump(
    portfolio_features_5d,
    os.path.join(models_folder, "equity_features_portfolio5d.pkl")
)

# ===== PORTFOLIO 21D RETURN MODEL =====
print("\n=== TRAINING PORTFOLIO 21D RETURN MODEL ===")

portfolio_features_21d = (
    feature_cols +
    [
        "lag_ret_1d",
        "lag_ret_5d",
        "lag_port_ret_contrib_1d",
        "lag_port_ret_contrib_5d"
    ]
)

portfolio_features_21d = [
    f for f in portfolio_features_21d
    if f in df.columns
]

target_21d = "port_ret_contrib_21d"

ml_21d = df[
    portfolio_features_21d + [target_21d]
].copy()

ml_21d = (
    ml_21d
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

X = ml_21d[portfolio_features_21d].values
y = ml_21d[target_21d].values

split_idx = int(len(ml_21d) * 0.80)
val_idx = int(split_idx * 0.90)

X_tr, y_tr = X[:val_idx], y[:val_idx]
X_val, y_val = X[val_idx:split_idx], y[val_idx:split_idx]
X_te, y_te = X[split_idx:], y[split_idx:]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_te)

watchlist = [
    (dtrain, "train"),
    (dval, "val")
]

model_21d_port = xgb.train(
    params_port,
    dtrain,
    num_boost_round=4000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=200
)

pred_21d_port = model_21d_port.predict(dtest)

print(
    "PORTFOLIO 21D - MSE:",
    mean_squared_error(y_te, pred_21d_port),
    "R2:",
    r2_score(y_te, pred_21d_port)
)

model_21d_port.save_model(
    os.path.join(models_folder, "equity_model_portfolio21d_xgb.json")
)

joblib.dump(
    portfolio_features_21d,
    os.path.join(models_folder, "equity_features_portfolio21d.pkl")
)

print("\n=== ALL 15 MODELS TRAINED AND SAVED SUCCESSFULLY! ===")
print("\nModels saved in:", models_folder)
print("\nModel Summary:")
print("1. 1-Day Return Model")
print("2. 5-Day Return Model")
print("3. 21-Day Return Model")
print("4. Volatility Forecast Model")
print("5. Downside Risk Model")
print("6. VaR Model")
print("7. Factor Return Model")
print("8. Sector Rotation Model")
print("9. Macro Regime Model")
print("10. Portfolio VaR 1D Model")
print("11. Portfolio VaR 5D Model")
print("12. Portfolio VaR 21D Model")
print("13. Portfolio 1D Return Model")
print("14. Portfolio 5D Return Model")
print("15. Portfolio 21D Return Model")