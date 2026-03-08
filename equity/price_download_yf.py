"""
Download daily OHLCV data for all S&P 500 tickers from Yahoo Finance.
"""

import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def download_sp500_prices(
    input_csv: str,
    output_csv: str,
    start_date: str = "2023-10-01",
    end_date: str = "2025-09-30"
):

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load tickers
    tickers = pd.read_csv(input_csv)["ticker"].dropna().unique().tolist()
    print("Loaded", len(tickers), "tickers from", input_csv)

    # Download daily OHLCV data
    print("Downloading", len(tickers), "tickers from", start_date, "to", end_date)

    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True
    )

    # Flatten multi-index
    print("Formatting data")
    records = []

    for t in tqdm(tickers, desc="Processing", ncols=90):
        try:
            if t not in data.columns.get_level_values(0):
                continue
            sub = (
                data[t]
                .reset_index()
                .rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
            )
            sub["ticker"] = t
            records.append(sub)
        except Exception:
            continue

    if not records:
        raise RuntimeError("No data downloaded.")

    final_df = pd.concat(records, ignore_index=True)
    final_df.to_csv(output_csv, index=False)

    print("Saved", len(final_df), "rows to", output_csv)
    return final_df



def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_csv = os.path.join(BASE_DIR, "data", "sp500_filtered.csv")
    output_csv = os.path.join(BASE_DIR, "data", "market_price_daily.csv")

    download_sp500_prices(input_csv=input_csv, output_csv=output_csv)


if __name__ == "__main__":
    main()
