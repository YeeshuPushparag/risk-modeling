import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

def fetch_macro_data(start_date="2022-01-01",
                     end_date="2026-02-28"):

    # --------------------------
    # PATH SETUP
    # --------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    OUT_FILE = os.path.join(OUTPUT_DIR, "macro_data.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------
    # ENV + FRED INIT
    # --------------------------
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("Missing FRED_API_KEY in .env")

    fred = Fred(api_key=api_key)

    print("Fetching macroeconomic data from FRED...")

    gdp = fred.get_series("GDP", start=start_date, end=end_date)          # Quarterly
    unrate = fred.get_series("UNRATE", start=start_date, end=end_date)    # Monthly
    cpi = fred.get_series("CPIAUCSL", start=start_date, end=end_date)     # Monthly
    fedfunds = fred.get_series("FEDFUNDS", start=start_date, end=end_date)# Monthly

    # --------------------------
    # Monthly Index
    # --------------------------
    monthly_index = pd.date_range(start=start_date, end=end_date, freq="ME")

    # GDP quarterly → monthly
    gdp_m = gdp.resample("ME").ffill().reindex(monthly_index).ffill()

    # Monthly series (take last value of month)
    unrate_m = unrate.resample("ME").last().reindex(monthly_index).ffill()
    cpi_m = cpi.resample("ME").last().reindex(monthly_index)
    fedfunds_m = fedfunds.resample("ME").last().reindex(monthly_index)

    # --------------------------
    # Final DataFrame
    # --------------------------
    df = pd.DataFrame({
        "date": monthly_index,
        "gdp": gdp_m.values,
        "unrate": unrate_m.values,
        "cpi": cpi_m.values,
        "fedfunds": fedfunds_m.values
    })

    # Save
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved macro_data -> {OUT_FILE}")


    return df


def main():
    fetch_macro_data()


if __name__ == "__main__":
    main()
