
import os
import pandas as pd
import numpy as np



# ============================================================
# 1️⃣ Load intermediate data
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
fx_file = os.path.join(DATA_DIR, "company_fx_exposure_with_interest_diff.csv")
fund_file = os.path.join(DATA_DIR, "corporate_fundamentals.csv")
macro_file = os.path.join(DATA_DIR, "macro_data.csv")
OUTPUT = os.path.join(DATA_DIR, "final", "fx.csv")
fx = pd.read_csv(fx_file)
fund = pd.read_csv(fund_file)
macro = pd.read_csv(macro_file)

fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
fund["date"] = pd.to_datetime(fund["date"], errors="coerce")

print(f"Loaded FX ({len(fx)} rows) and fundamentals ({len(fund)} rows)")

# ============================================================
# 2️⃣ FX metrics: returns, vol, momentum, Sharpe
# ============================================================
fx = fx.sort_values(["ticker", "date"])
fx["fx_return"] = fx.groupby("currency_pair")["fx_rate"].pct_change()
fx["fx_volatility_20d"] = fx.groupby("currency_pair")["fx_return"].transform(lambda x: x.rolling(20).std())
fx["fx_volatility_30d"] = fx.groupby("currency_pair")["fx_return"].transform(lambda x: x.rolling(30).std())

def downside_risk(x, window=20):
    neg = x.copy()
    neg[neg > 0] = 0
    return neg.rolling(window).std()

fx["downside_risk_20d"] = fx.groupby("currency_pair")["fx_return"].transform(lambda x: downside_risk(x, 20))
fx["momentum_5d"] = fx.groupby("currency_pair")["fx_rate"].transform(lambda x: x.pct_change(5))
fx["momentum_20d"] = fx.groupby("currency_pair")["fx_rate"].transform(lambda x: x.pct_change(20))
fx["sharpe_ratio_20d"] = fx["fx_return"] / fx["fx_volatility_20d"]
fx.dropna(subset=["fx_return"], inplace=True)
print("Rolling metrics computed")

# ============================================================
# 3️⃣ Merge fundamentals -> compute position size
# ============================================================
fx["Quarter_End"] = fx["date"].dt.to_period("Q").dt.end_time.dt.normalize()
merged = pd.merge(
    fx,
    fund[["ticker", "date", "revenue"]],
    left_on=["ticker", "Quarter_End"],
    right_on=["ticker", "date"],
    how="inner"
)
TRADING_DAYS_PER_Q = 63

merged["position_size"] = (
    merged["revenue"] / TRADING_DAYS_PER_Q
) * merged["foreign_revenue_ratio"]

merged.drop(["revenue", "Quarter_End", "date_y"], axis=1, inplace=True)
merged.rename(columns={"date_x": "date"}, inplace=True)
print(f"Position sizes merged, {len(merged)} rows")

# ============================================================
# 4️⃣ Risk, VaR, Hedge, PnL (EXACTLY LIKE ORIGINAL)
# ============================================================
df = merged.sort_values(["currency_pair", "date"]).reset_index(drop=True)

# 1) Volatility Setup: EWMA blend of 20d/30d for smoother, realistic signal
vol_20 = df.get("fx_volatility_20d", df.get("fx_volatility", np.nan))
vol_30 = df.get("fx_volatility_30d", df.get("fx_volatility", np.nan))
df["fx_volatility_blend"] = 0.7 * vol_20.fillna(0) + 0.3 * vol_30.fillna(0)

def ewma_group(s, span=10):
    return s.ewm(span=span, adjust=False).mean()

df["fx_volatility"] = df.groupby("currency_pair")["fx_volatility_blend"].transform(
    lambda x: ewma_group(x, span=10)
)

# 2) Carry Term & Carry-Adjusted Return
TRADING_DAYS = 252
df["carry_daily"] = (df["interest_diff"] / 100) / TRADING_DAYS
df["return_carry_adj"] = df["fx_return"] + df["carry_daily"]

# 3) Hedge Ratio (Economically Correct)
def minmax_grp(s):
    vmin, vmax = s.min(), s.max()
    rng = (vmax - vmin) if vmax > vmin else 1.0
    return (s - vmin) / rng

# Scale volatility per pair
df["vol_scaled"] = df.groupby("currency_pair")["fx_volatility"].transform(minmax_grp)

# Scale |interest_diff| globally
abs_int = abs(df["interest_diff"])
imin, imax = abs_int.min(), abs_int.max()
irange = (imax - imin) if imax > imin else 1.0
df["int_scaled"] = 1 - (abs_int - imin) / irange

# Hedge ratio components
BASE_HEDGE = 0.10
W_VOL = 0.75
W_CARRY = 0.15

df["hedge_ratio_raw"] = BASE_HEDGE + W_VOL * df["vol_scaled"] + W_CARRY * df["int_scaled"]

# Logistic smoothing
med_vol = df["vol_scaled"].median()
sigmoid = 1 / (1 + np.exp(-6 * (df["vol_scaled"] - med_vol)))
df["hedge_ratio"] = np.clip(0.5 * sigmoid + 0.5 * df["hedge_ratio_raw"], 0, 1)

# 4) Exposure, Risk Metrics, and Liquidity Proxy
df["position_size"] = df["position_size"].fillna(0.0)
df["exposure_amount"] = df["position_size"] * (1 - df["hedge_ratio"])

# VaR calculations
Z_95, Z_99 = 1.65, 2.33
df["VaR_95"] = df["position_size"] * df["fx_volatility"] * Z_95
df["VaR_99"] = df["position_size"] * df["fx_volatility"] * Z_99
df["value_at_risk"] = df["VaR_99"]  # keep backward compatibility

# Liquidity proxy
df["volume"] = 0.25 * df["position_size"]

# 5) PnL Decomposition
# FX mark-to-market PnL
df["fx_pnl"] = df["exposure_amount"] * df["fx_return"]

# Carry PnL
df["carry_pnl"] = df["exposure_amount"] * df["carry_daily"]

# Total realized/expected daily PnL
df["total_pnl"] = df["fx_pnl"] + df["carry_pnl"]

# Expected (carry-adjusted) PnL — same as total for forecasted models
df["expected_pnl"] = df["total_pnl"]

# Sharpe-like ratio (return per unit risk per capital)
df["sharpe_like_ratio"] = df["total_pnl"] / (
    df["position_size"] * df["fx_volatility"].replace(0, np.nan)
)

# 6) Warm-up Handling
df["is_warmup"] = df[["fx_volatility_20d", "fx_volatility_30d"]].isna().any(axis=1)
df_prod = df[~df["is_warmup"]].copy()

# 7) Final Columns & Export (KEEP ONLY SPECIFIED COLUMNS)
final_cols = [
    "ticker","sector","industry","currency_pair","foreign_revenue_ratio","date",
    "fx_rate","fx_return","fx_volatility_20d","fx_volatility_30d","fx_volatility",
    "interest_diff","carry_daily","return_carry_adj",
    "position_size","hedge_ratio","exposure_amount",
    "fx_pnl","carry_pnl","total_pnl","expected_pnl",
    "VaR_95","VaR_99","value_at_risk",
    "volume","sharpe_like_ratio","is_warmup"
]

# Keep only columns that exist
available_cols = [c for c in final_cols if c in df_prod.columns]
df_final = df_prod[available_cols]

print(f"Risk & PnL metrics calculated, {len(df_final)} rows after warmup removal")

# ============================================================
# 5️⃣ Merge Macro Data
# ============================================================
df_final["date"] = pd.to_datetime(df_final["date"], errors="coerce")
macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
df_final["mm_yy"] = df_final["date"].dt.strftime("%m-%y")
macro["mm_yy"] = macro["date"].dt.strftime("%m-%y")
macro = macro.drop(columns=["date"])

merged_macro = pd.merge(df_final, macro, on="mm_yy", how="left").drop(columns=["mm_yy"])

merged_macro.to_csv(OUTPUT, index=False)

print(f"Final FX + Macro dataset saved -> {OUTPUT}")
print(f"Rows: {len(merged_macro)}, Columns: {len(merged_macro.columns)}")
print(f"Warmup dropped: {df['is_warmup'].sum()}")
print(merged_macro.head(3))