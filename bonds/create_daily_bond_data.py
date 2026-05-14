import os
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BONDS_INPUT = os.path.join(DATA_DIR, "synthetic_bond.csv")
MACRO_DATA = os.path.join(DATA_DIR, "macro_data.csv")
FINAL_OUTPUT = os.path.join(DATA_DIR, "final", "bonds.csv")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

RECOVERY_RATE = 0.40

def create_daily_bond_data():
    """Create daily bond data with market dynamics and macro integration"""
    
    # === LOAD BASE BOND DATA ===
    base = pd.read_csv(BONDS_INPUT)
    print(f"Loaded base bond data with {len(base)} bonds")
    
    # === CREATE DAILY TIME SERIES ===
    start_date, end_date = "2024-07-01", "2025-09-30"
    dates = pd.date_range(start_date, end_date, freq="B")
    n_days = len(dates)

    daily = base.loc[base.index.repeat(n_days)].copy()
    daily["date"] = np.tile(dates, len(base))
    print(f"Created daily series: {len(daily)} rows")

    # === VOLATILITY BY RATING ===
    vol_map = {"AAA":2,"AA+":3,"AA":4,"A+":5,"A":6,"A-":8,"BBB+":10,"BBB":12,"BBB-":15,"BB+":20,"BB":25,"B":35,"CCC":50}
    daily["vol"] = daily["credit_rating"].map(vol_map).fillna(10)/100.0

    # === MACRO DRIVER: 10Y Treasury ===
    fred = Fred(api_key=FRED_API_KEY)
    dgs10 = fred.get_series("DGS10", observation_start=start_date, observation_end=end_date)\
                .to_frame("DGS10").reset_index().rename(columns={"index": "date"})
    biz = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="B")})
    dgs10 = biz.merge(dgs10, on="date", how="left").ffill()
    dgs10["DGS10_ma"] = dgs10["DGS10"].rolling(20, min_periods=1).mean()
    dgs10["dgs10_anom"] = dgs10["DGS10"] - dgs10["DGS10_ma"]
    daily = daily.merge(dgs10[["date","DGS10","dgs10_anom"]], on="date", how="left")
    print("Fetched Treasury yield data")

    # === RATING BETA FOR YIELD ADJUSTMENT ===
    rating_beta = {"AAA":0.6,"AA+":0.7,"AA":0.8,"A+":0.9,"A":1.0,"A-":1.1,"BBB+":1.2,"BBB":1.3,
                   "BBB-":1.4,"BB+":1.6,"BB":1.8,"B":2.0,"CCC":2.5}
    daily["rating_beta"] = daily["credit_rating"].map(rating_beta).fillna(1.0)

    # === MARKET DYNAMICS ===
    np.random.seed(42)
    
    # Adjust benchmark yield based on Treasury anomalies
    daily["benchmark_yield"] = (
        daily["benchmark_yield"] +
        daily["rating_beta"] * daily["dgs10_anom"] * 0.8 +
        np.random.normal(0, 0.01, len(daily))
    )

    # Credit spread shocks with mean reversion
    spread_shock = np.random.normal(0, 1, len(daily)) * daily["vol"]
    daily["credit_spread"] = (
        daily["credit_spread"] +
        spread_shock -
        0.10 * (daily["credit_spread"] - daily.groupby("ticker")["credit_spread"].transform("mean"))
    ).clip(lower=0)

    # Recalculate yields and prices
    daily["corporate_yield"] = daily["benchmark_yield"] + daily["credit_spread"]
    daily["yield_to_maturity"] = daily["corporate_yield"]

    r = daily["corporate_yield"] / 100
    c = daily["coupon_rate"] * 100
    daily["bond_price"] = (
        c * (1 - (1 + r) ** (-daily["maturity_years"])) / r + 
        100 / (1 + r) ** daily["maturity_years"]
    )

    # === UPDATE CREDIT METRICS ===
    daily["implied_hazard"] = daily["credit_spread"] / 100 / (1 - RECOVERY_RATE)
    daily["implied_pd_annual"] = 1 - np.exp(-daily["implied_hazard"])
    daily["implied_pd_multi_year"] = 1 - np.exp(-daily["implied_hazard"] * daily["maturity_years"])

    def band(pd_a):
        if pd_a < 0.001: return "AAA"
        if pd_a < 0.002: return "AA+"
        if pd_a < 0.003: return "AA"
        if pd_a < 0.005: return "A+"
        if pd_a < 0.010: return "A"
        if pd_a < 0.020: return "A-"
        if pd_a < 0.050: return "BBB"
        if pd_a < 0.100: return "BB"
        if pd_a < 0.200: return "B"
        return "CCC"

    daily["implied_rating"] = daily["implied_pd_annual"].apply(band)
    daily["market_synthetic_score"] = (100 - (daily["corporate_yield"] - daily["benchmark_yield"]) * 20).clip(0, 100)

    # === SELECT FINAL COLUMNS ===
    cols = [
        "bond_id", "ticker","sector","industry","credit_rating","coupon_rate",
        "issue_date","maturity_date","maturity_years","date",
        "benchmark_yield","corporate_yield","credit_spread","bond_price",
        "yield_to_maturity","implied_hazard","implied_pd_annual",
        "implied_pd_multi_year","implied_rating","market_synthetic_score", "issue_size",
        "face_value_per_bond", "units_issued", "units_outstanding", "market_value",
    ]
    daily_data = daily[cols].reset_index(drop=True)

    # === MERGE MACRO DATA ===
    if os.path.exists(MACRO_DATA):
        macro_df = pd.read_csv(MACRO_DATA, parse_dates=["date"])
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        daily_data["mm_yy"] = daily_data["date"].dt.strftime("%m-%y")
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        macro_df["mm_yy"] = macro_df["date"].dt.strftime("%m-%y")
        macro_df = macro_df.drop(columns=["date"])
        
        merged_df = pd.merge(daily_data, macro_df, on="mm_yy", how="left").drop(columns=["mm_yy"])
        print("Merged with macro data")
    else:
        merged_df = daily_data
        print("Macro data not found, saving bond data only")

    return merged_df

def main():
    print("Creating daily bond data with market dynamics...")
    final_df = create_daily_bond_data()
    
    # Save final output
    os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    
    print(f"Final daily bond data saved -> {FINAL_OUTPUT}")
    print(f"Rows: {len(final_df)} | Columns: {len(final_df.columns)}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(final_df.head(3))

if __name__ == "__main__":
    main()