import os
import pandas as pd
import numpy as np

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    # === LOAD DATA ===
    loans = pd.read_csv(os.path.join(DATA_DIR, "loan_enriched_fx_bonds_commod_derivatives_collateral.csv"))
    macro = pd.read_csv(os.path.join(DATA_DIR, "macro_data.csv"))

    # === MERGE WITH MACRO ===
    print("Merging with macro data...")
    loans["month_year"] = pd.to_datetime(loans["date"]).dt.to_period("M").astype(str)
    macro["month_year"] = pd.to_datetime(macro["date"]).dt.to_period("M").astype(str)
    
    df = loans.merge(macro.drop(columns=["date"]), on="month_year", how="left").drop(columns=["month_year"])
    print(f"After macro merge: {df.shape}")

    # === PARSE DATES ===
    for c in ["date", "issue_date", "maturity_date"]:
        s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        m = s.isna()
        if m.any():
            s[m] = pd.to_datetime(df.loc[m, c], format="%d-%m-%Y", errors="coerce")
        df[c] = s

    # === BASIC SCALING ===
    if df["coupon_rate"].abs().median() >= 1:
        df["coupon_rate"] /= 100.0
    df["spread_rate"] = df["spread_bps"] / 10000.0

    # === TIME METRICS ===
    df["loan_age_months"] = ((df["date"] - df["issue_date"]).dt.days / 30).clip(lower=0)
    df["time_to_maturity_months"] = ((df["maturity_date"] - df["date"]).dt.days / 30).clip(lower=0)

    # === INTEREST INCOME ===
    df["interest_rate_monthly"] = (df["coupon_rate"] + df["spread_rate"]) / 12.0
    df["interest_income"] = df["notional_usd"] * df["interest_rate_monthly"]

    # === COLLATERAL COVERAGE ===
    if "collateral_value" not in df.columns:
        df["collateral_value"] = df["exposure_before_collateral"] * df["collateral_ratio"]
    df["exposure_pct_collateralized"] = (
        df["collateral_value"] / df["exposure_before_collateral"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # === MACRO STRESS ===
    for col in ["unrate", "fedfunds"]:
        mu, sd = df[col].mean(), df[col].std(ddof=0)
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0
    df["macro_stress_score"] = df["unrate_z"] + df["fedfunds_z"]

    # === VOLATILITY INDEX ===
    df = df.sort_values(["loan_id", "date"])
    WINDOW = 3
    df["cs_roll_std"] = df.groupby("loan_id")["credit_spread"].transform(lambda s: s.rolling(WINDOW, min_periods=2).std())
    df["fxv_roll_std"] = df.groupby("loan_id")["fx_volatility"].transform(lambda s: s.rolling(WINDOW, min_periods=2).std())
    df["cmd_roll_std"] = df.groupby("loan_id")["vol_20d"].transform(lambda s: s.rolling(WINDOW, min_periods=2).std())

    for col in ["cs_roll_std", "fxv_roll_std", "cmd_roll_std"]:
        mu, sd = df[col].mean(), df[col].std(ddof=0)
        df[col] = (df[col] - mu) / sd if sd > 0 else 0

    df["volatility_index"] = df[["cs_roll_std", "fxv_roll_std", "cmd_roll_std"]].mean(axis=1)

    # === RATIOS ===
    df["credit_spread_ratio"] = (df["credit_spread"] / df["yield_to_maturity"]).replace([np.inf, -np.inf], np.nan)
    df["profitability_ratio"] = (df["pnl"] / df["exposure_before_collateral"]).replace([np.inf, -np.inf], np.nan)
    df["utilization_ratio"] = (df["net_exposure"] / df["notional_usd"]).replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # === FINAL FEATURE OUTPUT ===
    cols = [
        "loan_id","ticker","sector","industry","currency","date","issue_date","maturity_date",
        "rate_type","coupon_rate","spread_bps","spread_rate","notional_usd","credit_rating",
        "credit_spread","yield_to_maturity","fx_rate","fx_volatility","carry_daily","close","vol_20d",
        "gdp","unrate","cpi","fedfunds","loan_age_months","time_to_maturity_months","interest_income",
        "exposure_pct_collateralized","macro_stress_score","volatility_index",
        "credit_spread_ratio","profitability_ratio","utilization_ratio","counterparty","funding_cost","liquidity_score"
    ]

    # Keep only columns that exist in the dataframe
    available_cols = [c for c in cols if c in df.columns]
    df_final = df[available_cols]
    
    output_path = os.path.join(DATA_DIR, 'final', "loans.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"Loan features final ready: {df_final.shape}")
    print(f"Output file: {output_path}")
    print(f"Columns used: {len(available_cols)} out of {len(cols)}")

if __name__ == "__main__":
    main()