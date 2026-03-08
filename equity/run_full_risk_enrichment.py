import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from math import sqrt

# ============================================================
# PATHS  (CHANGE ONLY IF YOU WANT)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINAL_MERGED = os.path.join(BASE_DIR, "data", "final_market_macro_fundamentals.csv")
FINAL_13F    = os.path.join(BASE_DIR, "data", "final_13f.csv")
OUTPUT_CSV   = os.path.join(BASE_DIR, "data", "final", "equity.csv")

# CONSTANTS
Z_95 = norm.ppf(0.05)
Z_99 = norm.ppf(0.01)
ALPHA_95 = 0.95
ALPHA_99 = 0.99

# Minimum values to prevent zeros
MIN_VOLATILITY = 0.01  # 1% minimum daily volatility
MIN_VaR_95 = 0.005     # 0.5% minimum VaR 95%
MIN_VaR_99 = 0.01      # 1% minimum VaR 99%
MIN_CVaR_95 = 0.007    # 0.7% minimum CVaR 95%
MIN_CVaR_99 = 0.015    # 1.5% minimum CVaR 99%


# ============================================================
# 1. LOAD AND MERGE 13F HOLDINGS + FINAL MARKET DATA
# ============================================================
def load_and_merge():
    print("Loading 13F aggregated CSV...")
    hold = pd.read_csv(FINAL_13F)
    hold["date"] = pd.to_datetime(hold["date"])

    print("Loading merged market-macro-fundamentals...")
    mkt = pd.read_csv(FINAL_MERGED)
    mkt["date"] = pd.to_datetime(mkt["date"])

    # Build QUARTER key
    hold["QUARTER"] = hold["date"].dt.to_period("Q")
    mkt["QUARTER"] = mkt["date"].dt.to_period("Q")

    # Drop hold.date to avoid duplicate
    hold_no_date = hold.drop(columns=["date"])

    print("Merging 13F with daily market data...")
    merged = pd.merge(
        hold_no_date,
        mkt,
        on=["ticker", "QUARTER"],
        how="inner"
    )

    merged = merged.drop(columns=["QUARTER"])
    merged = merged.sort_values(["asset_manager", "ticker", "date"]).reset_index(drop=True)

    print("Merged rows:", len(merged))
    print("Unique tickers:", merged["ticker"].nunique())
    return merged


# ============================================================
# 2. FINANCIAL ENRICHMENT
# ============================================================
def financial_enrichment(df):
    df["interest_coverage_proxy"] = df.apply(
        lambda x: x["ebitda"] / (0.05 * x["totalDebt"])
        if x.get("totalDebt") and x["totalDebt"] > 0 else np.nan,
        axis=1
    )

    df["maturity_proxy"] = df.apply(
        lambda x: x["longTermDebt"] / x["totalDebt"]
        if x.get("totalDebt") and x["totalDebt"] > 0 else np.nan,
        axis=1
    )

    def altman_z(row):
        if row["totalAssets"] <= 0 or row["totalDebt"] <= 0:
            return np.nan
        wc_ta = (row["totalAssets"] - row["totalDebt"]) / row["totalAssets"]
        re_ta = row["netIncome"] / row["totalAssets"]
        ebit_ta = (row["ebitda"] - 0.05 * row["totalDebt"]) / row["totalAssets"]
        mve_tl = row["marketCap"] / row["totalDebt"]
        s_ta = row["revenue"] / row["totalAssets"]
        return 1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mve_tl + 1.0*s_ta

    df["Altman_Z"] = df.apply(altman_z, axis=1)

    df["free_cash_flow"] = df.apply(
        lambda x: x["ebitda"] - 0.10 * x["revenue"]
        if pd.notnull(x["ebitda"]) and pd.notnull(x["revenue"]) else np.nan,
        axis=1
    )

    df["cash_and_equivalents"] = df.apply(
        lambda x: x["totalAssets"] - x["totalDebt"] - 0.5*x["revenue"]
        if pd.notnull(x["totalAssets"]) and pd.notnull(x["totalDebt"]) else np.nan,
        axis=1
    )

    def rate(z):
        if pd.isna(z): return None
        elif z >= 8.15: return "AAA"
        elif z >= 7.0: return "AA"
        elif z >= 5.85: return "A"
        elif z >= 4.2: return "BBB"
        elif z >= 3.0: return "BB"
        elif z >= 1.8: return "B"
        elif z >= 1.23: return "CCC"
        elif z >= 0.8: return "CC"
        elif z >= 0.5: return "C"
        else: return "D"

    df["credit_rating"] = df["Altman_Z"].apply(rate)
    df["mtm_value"] = df["close"] * df["Shares"]

    print("Financial enrichment complete.")
    return df

# ============================================================
# 3. RISK ENRICHMENT
# ============================================================
def risk_enrichment(df):
    
    numeric_cols = [
        "total_value","total_percent","Shares","volume", "mtm_value",
        "daily_return","vol_5d","vol_20d","volatility_21d","downside_risk",
        "sharpe_ratio","sortino_ratio","beta",
        "avg_volume_10d",
        "gdp","unrate","cpi","fedfunds",
        "totalDebt","shortTermDebt","longTermDebt","totalAssets",
        "ebitda","revenue","netIncome","marketCap","dividendYield",
        "Altman_Z","free_cash_flow","cash_and_equivalents",
    ]

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # === SECURITY-LEVEL DAILY PNL ============================================
    df["daily_pnl"] = df["mtm_value"] * df["daily_return"]

    total_grp = (
        df.groupby(["asset_manager","date"])["mtm_value"]
          .transform("sum").replace(0, np.nan)
    )
    df["portfolio_weight"] = (df["mtm_value"] / total_grp).fillna(0.0)

    sec = (
        df.groupby(["asset_manager","date","sector"])["portfolio_weight"]
          .sum().reset_index().rename(columns={"portfolio_weight":"sector_exposure"})
    )
    ind = (
        df.groupby(["asset_manager","date","industry"])["portfolio_weight"]
          .sum().reset_index().rename(columns={"portfolio_weight":"industry_exposure"})
    )

    df = df.merge(sec, on=["asset_manager","date","sector"], how="left")
    df = df.merge(ind, on=["asset_manager","date","industry"], how="left")

    df["beta"] = df["beta"].fillna(0.0)
    df["beta_weighted"] = df["portfolio_weight"] * df["beta"]

    sec_beta = (
        df.groupby(["asset_manager","date","sector"])["beta_weighted"]
          .sum().reset_index().rename(columns={"beta_weighted":"sector_beta_weighted"})
    )
    df = df.merge(sec_beta, on=["asset_manager","date","sector"], how="left")


    # FIXED: Ensure daily_sigma has reasonable minimum values
    df["daily_sigma"] = (
        df["volatility_21d"]
          .fillna(df["vol_20d"])
          .fillna(df["vol_5d"])
          .fillna(MIN_VOLATILITY)  # Use minimum instead of 0
          .astype(float)
    )
    df["daily_sigma"] = df["daily_sigma"].replace(0, MIN_VOLATILITY)  # Replace any remaining zeros
    
    df["daily_mu"] = df["daily_return"].astype(float).fillna(0.0)

    # FIXED: VaR calculations with minimum values
    df["daily_VaR_95"] = (-(df["daily_mu"] + Z_95 * df["daily_sigma"])).clip(lower=MIN_VaR_95)
    df["daily_VaR_99"] = (-(df["daily_mu"] + Z_99 * df["daily_sigma"])).clip(lower=MIN_VaR_99)

    def parametric_es(mu, sigma, alpha):
        # FIXED: Always ensure reasonable minimum values
        if sigma is None or np.isnan(sigma) or sigma <= 0:
            sigma = MIN_VOLATILITY
        
        z = norm.ppf(1 - alpha)
        pdf = norm.pdf(z)
        es = -mu + sigma * (pdf / alpha)
        
        # Apply minimum values based on confidence level
        if alpha == ALPHA_95:
            return max(MIN_CVaR_95, float(es))
        else:  # ALPHA_99
            return max(MIN_CVaR_99, float(es))

    df["daily_CVaR_95"] = df.apply(lambda r: parametric_es(r["daily_mu"], r["daily_sigma"], ALPHA_95), axis=1)
    df["daily_CVaR_99"] = df.apply(lambda r: parametric_es(r["daily_mu"], r["daily_sigma"], ALPHA_99), axis=1)

    def port_metrics(g):
        w = g["portfolio_weight"].fillna(0).astype(float).values
        sigma = g["daily_sigma"].fillna(MIN_VOLATILITY).astype(float).values  # FIXED: Use minimum vol
        mu = (g["portfolio_weight"].fillna(0) * g["daily_mu"].fillna(0)).sum()
        
        # FIXED: Ensure minimum portfolio volatility
        port_vol = max(MIN_VOLATILITY, sqrt(np.sum((w * sigma)**2)))
        
        # FIXED: Apply minimum values to all portfolio risk metrics
        return pd.Series({
            "daily_portfolio_ex_ante_volatility": port_vol,
            "daily_portfolio_VaR_95": max(MIN_VaR_95, -(mu + Z_95*port_vol)),
            "daily_portfolio_VaR_99": max(MIN_VaR_99, -(mu + Z_99*port_vol)),
            "daily_portfolio_CVaR_95": max(MIN_CVaR_95, parametric_es(mu, port_vol, ALPHA_95)),
            "daily_portfolio_CVaR_99": max(MIN_CVaR_99, parametric_es(mu, port_vol, ALPHA_99)),
            "daily_portfolio_mu": mu
        })

    p = df.groupby(["asset_manager","date"]).apply(port_metrics, include_groups=False).reset_index()
    df = df.merge(p, on=["asset_manager","date"], how="left")


    def top5_hhi(g):
        w = g["portfolio_weight"].fillna(0)
        top5 = float(w.nlargest(5).sum())
        sector_hhi = float(np.sum(g.groupby("sector")["portfolio_weight"].sum() ** 2))
        return pd.Series({"top_5_exposure": top5, "HHI_sector": sector_hhi})

    hh = df.groupby(["asset_manager","date"]).apply(top5_hhi, include_groups=False).reset_index()
    df = df.merge(hh, on=["asset_manager","date"], how="left")
    df["diversification_score"] = df["HHI_sector"].apply(lambda x: 1/x if (pd.notna(x) and x > 0) else np.nan)

    df["avg_volume_10d"] = df["avg_volume_10d"].fillna(0.0)
    df["Amihud_illiquidity"] = df["daily_return"].abs() / df["avg_volume_10d"].replace(0, np.nan)

    df["turnover_ratio"] = np.where(
        (pd.notna(df["Shares"])) & (df["Shares"] > 0),
        df["volume"] / df["Shares"],
        np.nan
    )
    
    def _minmax(x):
        xm, xM = x.min(), x.max()
        if pd.isna(xm) or pd.isna(xM) or xM == xm:
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - xm) / (xM - xm)
        
    df["liquidity_risk_score"] = (
        df.groupby(["asset_manager","date"])["Amihud_illiquidity"].transform(_minmax)
    )
        # Liquidity risk score (minmax)
    df["liquidity_risk_score"] = df.groupby(["asset_manager","date"])["Amihud_illiquidity"] \
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)
    df["Amihud_illiquidity"] = df["Amihud_illiquidity"].fillna(0)
    df["liquidity_risk_score"] = df["liquidity_risk_score"].fillna(0)
    df["turnover_ratio"] = df["turnover_ratio"].fillna(0)
    df["sector_exposure_pct"] = df["sector_exposure"]

    # === PORTFOLIO-LEVEL DAILY PNL ===========================================
    portfolio_pnl = (
        df.groupby(["asset_manager","date"])["daily_pnl"]
          .sum()
          .reset_index()
          .rename(columns={"daily_pnl": "portfolio_daily_pnl"})
    )
    
    df = df.merge(portfolio_pnl, on=["asset_manager","date"], how="left")

    return df

# ============================================================
# MAIN
# ============================================================
def main():

    df = load_and_merge()

    df = financial_enrichment(df)
    df = risk_enrichment(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved enriched portfolio dataset ->", OUTPUT_CSV)
    print("Final row count:", len(df))


if __name__ == "__main__":
    main()
