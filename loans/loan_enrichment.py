import os
import pandas as pd
import numpy as np

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def fast_parse_dates(series):
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    mask = s.isna()
    if mask.any():
        s[mask] = pd.to_datetime(series[mask], format="%d-%m-%Y", errors="coerce")
    return s

def main():
    # === LOAD DATA ===
    loans = pd.read_csv(os.path.join(DATA_DIR, "loan_synthetic_base.csv"))
    fx = pd.read_csv(os.path.join(DATA_DIR, "final", "fx.csv"))
    bonds = pd.read_csv(os.path.join(DATA_DIR, "final", "bonds.csv"))
    commod = pd.read_csv(os.path.join(DATA_DIR, "final", "commodities.csv"))
    deriv = pd.read_csv(os.path.join(DATA_DIR, "final", "derivatives.csv"))
    collateral = pd.read_csv(os.path.join(DATA_DIR, "final", "collateral.csv"))

    # === PARSE DATES ===
    loans["date"] = fast_parse_dates(loans["date"])
    fx["date"] = fast_parse_dates(fx["date"])
    bonds["date"] = fast_parse_dates(bonds["date"])
    commod["date"] = fast_parse_dates(commod["date"])
    deriv["date"] = fast_parse_dates(deriv["date"])
    collateral["date"] = fast_parse_dates(collateral["date"])

    # === MONTH-YEAR KEYS ===
    loans["month_year"] = loans["date"].dt.to_period("M")
    fx["month_year"] = fx["date"].dt.to_period("M")
    bonds["month_year"] = bonds["date"].dt.to_period("M")
    commod["month_year"] = commod["date"].dt.to_period("M")
    deriv["month_year"] = deriv["date"].dt.to_period("M")
    collateral["month_year"] = collateral["date"].dt.to_period("M")

    # === AGGREGATIONS ===
    fx_month = fx.groupby(["ticker", "month_year"]).agg({
        "fx_rate": "mean",
        "fx_volatility": "mean",
        "carry_daily": "mean"
    }).reset_index()

    bonds_month = bonds.groupby(["ticker", "month_year"]).agg({
        "credit_spread": "mean",
        "yield_to_maturity": "mean",
        "credit_rating": lambda x: x.mode()[0] if len(x.mode()) else None
    }).reset_index()

    commod_month = commod.groupby(["sector", "month_year"]).agg({
        "close": "mean",
        "vol_20d": "mean"
    }).reset_index()

    deriv_month = (
        deriv.groupby(["ticker", "month_year"])
             .agg({
                 "notional": "mean",
                 "exposure_before_collateral": "mean",
                 "collateral_value": "mean",
                 "net_exposure": "mean",
                 "collateral_ratio": "mean",
                 "margin_call_flag": lambda x: 1 if (x == 1).any() else 0,
                 "pnl": 'mean'
             })
             .reset_index()
    )

    collat_month = (
        collateral.groupby(["ticker", "month_year"], as_index=False)
                  .agg({
                      "counterparty": lambda x: '|'.join(sorted(set(map(str, x)))[:3]),
                      "funding_cost": "mean",
                      "liquidity_score": "mean",
                      "margin_call_amount": "mean"
                  })
    )

    # === MERGES ===
    print("Merging FX data...")
    merged = (
        loans.merge(fx_month, how="inner", left_on=["ticker", "month_year"], right_on=["ticker", "month_year"])
             .merge(bonds_month, how="left", on=["ticker", "month_year"])
             .merge(commod_month, how="left", on=["sector", "month_year"])
    )

    print("Merging derivatives data...")
    merged = merged.merge(deriv_month, how="left", left_on=["ticker", "month_year"], right_on=["ticker", "month_year"])

    print("Merging collateral data...")
    merged = merged.merge(collat_month, how="left", on=["ticker", "month_year"])

    # Clean up columns
    merged.drop(columns=["month_year"], inplace=True, errors="ignore")

    # Save final output
    output_path = os.path.join(DATA_DIR, "loan_enriched_fx_bonds_commod_derivatives_collateral.csv")
    merged.to_csv(output_path, index=False)

    print(f"Loan enrichment completed: {merged.shape}")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    main()