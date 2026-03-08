import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# =====================================================
# FIXED PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKET_IN = os.path.join(BASE_DIR, "data", "market_price_daily.csv")
MACRO_IN = os.path.join(BASE_DIR, "data", "macro_data.csv")
FUND_IN = os.path.join(BASE_DIR, "data", "corporate_fundamentals.csv")

OUTPUT_MERGED = os.path.join(BASE_DIR, "data", "final_market_macro_fundamentals.csv")
os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)


# =====================================================
# 1. FEATURE ENGINEERING
# =====================================================
def generate_market_features(df, risk_free_rate=0.05):

    results = []

    for ticker, sub in tqdm(df.groupby("ticker"), desc="Computing features", ncols=100):
        sub = sub.copy()

        # BASIC RETURNS
        sub["daily_return"] = sub["close"].pct_change()
        sub["log_return"] = np.log(sub["close"] / sub["close"].shift(1))

        # STRUCTURAL
        sub["high_low_spread"] = (sub["high"] - sub["low"]) / sub["open"]
        sub["close_open_diff"] = (sub["close"] - sub["open"]) / sub["open"]

        # VOLATILITY
        sub["vol_5d"] = sub["daily_return"].rolling(5, min_periods=2).std() * np.sqrt(252)
        sub["vol_20d"] = sub["daily_return"].rolling(20, min_periods=5).std() * np.sqrt(252)
        sub["volatility_21d"] = sub["daily_return"].rolling(21, min_periods=5).std()

        # MOVING AVG
        sub["ma_5"] = sub["close"].rolling(5, min_periods=1).mean()
        sub["ma_20"] = sub["close"].rolling(20, min_periods=1).mean()

        # MOMENTUM
        sub["momentum_5d"] = sub["close"].pct_change(5)
        sub["momentum_20d"] = sub["close"].pct_change(20)
        sub["vol_change"] = sub["vol_change"].fillna(0)
        # VOLUME
        sub["avg_volume_10d"] = sub["volume"].rolling(10, min_periods=1).mean()
        sub["vol_change"] = sub["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

        # RISK ADJUSTED
        sub["excess_return"] = sub["daily_return"] - (risk_free_rate / 252)
        sub["downside_risk"] = (
            sub["daily_return"]
            .where(sub["daily_return"] < 0)
            .rolling(21, min_periods=5)
            .std()
        )

        sub["sharpe_ratio"] = (
            sub["excess_return"].rolling(21, min_periods=5).mean()
            / sub["volatility_21d"]
        )
        sub["sortino_ratio"] = (
            sub["excess_return"].rolling(21, min_periods=5).mean()
            / sub["downside_risk"]
        )
        sub["downside_risk"] = sub["downside_risk"].fillna(0)
        sub["sortino_ratio"] = sub["sortino_ratio"].fillna(0)
        results.append(sub)

    final = pd.concat(results, ignore_index=True)
    final.replace([np.inf, -np.inf], np.nan, inplace=True)
    final.dropna(subset=["close"], inplace=True)

    return final


# =====================================================
# 2. MERGE MARKET + MACRO + FUNDAMENTALS
# =====================================================
def merge_all():

    # --- LOAD ---
    market = pd.read_csv(MARKET_IN)
    macro = pd.read_csv(MACRO_IN)
    fund = pd.read_csv(FUND_IN)

    for df in [market, macro, fund]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- FEATURE ENGINEERING (NO SAVING) ---
    market = market.sort_values(["ticker", "date"])
    market_feat = generate_market_features(market)

    # --- MARKET + MACRO MONTH MERGE ---
    market_feat["mm_yy"] = market_feat["date"].dt.to_period("M").astype(str)
    macro["mm_yy"] = macro["date"].dt.to_period("M").astype(str)

    merged = pd.merge(
        market_feat,
        macro.drop(columns=["date"]),
        on="mm_yy",
        how="inner"
    )

    # --- FUNDAMENTALS QUARTER MERGE ---
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    fund["quarter"] = fund["date"].dt.to_period("Q").astype(str)
    fund = fund[fund["totalDebt"] != 0] 
    merged = pd.merge(
        merged,
        fund.drop(columns=["date"]),
        on=["ticker", "quarter"],
        how="inner"
    )

    # CLEAN
    cols_with_nulls = [
        "vol_20d", "momentum_5d", "momentum_20d",
        "vol_change", "downside_risk", "volatility_21d",
        "sharpe_ratio", "sortino_ratio"
    ]
    merged = merged.dropna(subset=cols_with_nulls)

    merged.drop(columns=["mm_yy", "quarter"], inplace=True)
    merged.sort_values(["ticker", "date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # SAVE ONLY FINAL MERGED FILE
    merged.to_csv(OUTPUT_MERGED, index=False)
    
    return merged


# =====================================================
# MAIN
# =====================================================
def main():
    merged = merge_all()
    print("DONE — Final merged rows:", len(merged))
    print("Saved to:", OUTPUT_MERGED)


if __name__ == "__main__":
    main()
