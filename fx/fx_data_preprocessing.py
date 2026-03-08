import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    # ===== STEP 1: Create heuristic FX exposure =====
    INPUT = os.path.join(DATA_DIR, "corporate_fundamentals.csv")
    OUTPUT = os.path.join(DATA_DIR, "company_fx_exposure_with_interest_diff.csv")
    SYM = os.path.join(DATA_DIR, "unique_sym_sector_industry.csv")
    # Load fundamentals and create unique tickers
    df = pd.read_csv(INPUT)[["ticker", "sector", "industry"]]
    df_unique = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    df_unique.to_csv(SYM, index=False)
    # Map sectors to FX exposure
    sector_to_currency_pair = {
        "Technology": "USDEUR", "Healthcare": "USDCHF", "Consumer Cyclical": "USDEUR",
        "Financial Services": "USDJPY", "Consumer Defensive": "USDCAD", "Utilities": "USDGBP",
        "Basic Materials": "USDAUD", "Industrials": "USDCNY", "Real Estate": "USDJPY",
        "Energy": "USDCAD", "Communication Services": "USDEUR"
    }

    foreign_ratio = {
        "USDEUR": 0.45, "USDCHF": 0.30, "USDJPY": 0.25,
        "USDCAD": 0.33, "USDGBP": 0.10, "USDAUD": 0.40, "USDCNY": 0.35
    }

    df_unique["currency_pair"] = df_unique["sector"].map(sector_to_currency_pair)
    df_unique["foreign_revenue_ratio"] = df_unique["currency_pair"].map(foreign_ratio)
    fx_map = df_unique
    print(f"Created FX exposure mapping for {len(fx_map)} tickers")

    # ===== STEP 2: Download FX rates =====
    pairs = [p for p in fx_map["currency_pair"].dropna().unique() if p != "USDUSD"]
    print("Fetching FX rates:", pairs)
    yahoo_map = {p: p + "=X" for p in pairs}

    data = yf.download(list(yahoo_map.values()), start="2022-01-01", end="2025-09-30")

    # Flatten MultiIndex if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    # Keep only Close prices
    close_cols = [c for c in data.columns if "Close" in c]
    fx = data[close_cols].copy()

    # Rename columns to currency pairs
    for p, ysym in yahoo_map.items():
        for col in fx.columns:
            if ysym in col:
                fx.rename(columns={col: p}, inplace=True)

    # Reset index to make Date a column
    fx.reset_index(inplace=True)
    fx.rename(columns={"Date": "date"}, inplace=True)
    fx.ffill(inplace=True)
    print(f"Downloaded FX rates with {len(fx)} rows")

    # ===== STEP 3: Combine FX map + rates =====
    fx_long = fx.melt(id_vars=["date"], var_name="currency_pair", value_name="fx_rate")
    company_fx = fx_map.merge(fx_long, on="currency_pair", how="left")
    print(f"Combined FX map + rates: {len(company_fx)} rows")

    # ===== STEP 4: FRED interest rates =====
    fred = Fred(api_key=FRED_API_KEY)
    rate_series = {
        "USD": "FEDFUNDS", "EUR": "ECBDFR", "GBP": "IR3TIB01GBM156N",
        "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
        "JPY": "IR3TIB01JPM156N", "CHF": "IR3TIB01CHM156N", "CNY": "INTDSRCNM193N"
    }

    all_rates = {}
    for cur, sid in rate_series.items():
        try:
            data = fred.get_series(sid, observation_start="2022-01-01", observation_end="2025-09-30")
            if data is not None and len(data) > 0:
                all_rates[cur] = pd.DataFrame({"date": data.index, cur: data.values})
                print(f"{cur}: {len(data)} records")
        except Exception as e:
            print(f"{cur}: {e}")

    rates_df = None
    for cur, df_rates in all_rates.items():
        rates_df = df_rates if rates_df is None else pd.merge(rates_df, df_rates, on="date", how="outer")

    rates_df.sort_values("date", inplace=True)
    rates_df.ffill(inplace=True)
    rates_df.bfill(inplace=True)
    
    for cur in rate_series.keys():
        if cur != "USD" and cur in rates_df.columns:
            rates_df[f"USD{cur}"] = rates_df["USD"] - rates_df[cur]
    rates_df["USDUSD"] = 0
    print("Interest differentials calculated")

    # ===== STEP 5: Merge FX + interest =====
    rates_df["date"] = pd.to_datetime(rates_df["date"], errors="coerce")
    company_fx["date"] = pd.to_datetime(company_fx["date"], errors="coerce")
    
    rate_cols = [c for c in rates_df.columns if c.startswith("USD") and len(c) == 6]
    rates_long = rates_df.melt(id_vars=["date"], value_vars=rate_cols,
                            var_name="currency_pair", value_name="interest_diff")

    company_fx = company_fx.dropna(subset=["date", "currency_pair"]).sort_values("date")
    rates_long = rates_long.dropna(subset=["date", "currency_pair"]).sort_values("date")

    merged = pd.merge_asof(company_fx, rates_long, on="date", by="currency_pair", direction="backward")
    merged.to_csv(OUTPUT, index=False)
    print(f"Saved final FX exposure with interest diff to {OUTPUT}")

if __name__ == "__main__":
    main()