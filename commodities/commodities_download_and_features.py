
import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "commodities_daily.csv")

TICKERS = ["GC=F", "CL=F", "SI=F", "NG=F", "ZC=F"]  # gold, oil, silver, natgas, corn
START, END = "2024-01-01", "2025-09-30"

print(f"Fetching {len(TICKERS)} commodities from {START} to {END}...")

# === DOWNLOAD ===
data = yf.download(TICKERS, start=START, end=END, group_by="ticker", progress=False)
records = []

for tkr in tqdm(TICKERS, desc="Processing", ncols=100):
    df = data[tkr].reset_index()
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["commodity_symbol"] = tkr
    df = df.dropna(subset=["close"]).sort_values("date")

    # === FEATURES ===
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_20d"] = df["daily_return"].rolling(20, min_periods=5).std() * np.sqrt(252)
    df["VaR_95"] = df["daily_return"].rolling(60, min_periods=20).quantile(0.05)
    df["VaR_99"] = df["daily_return"].rolling(60, min_periods=20).quantile(0.01)
    df["pnl"] = df["close"].diff()

    records.append(df)

final = pd.concat(records, ignore_index=True).dropna(subset=["daily_return"])
final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")
final.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(final):,} rows across {final['commodity_symbol'].nunique()} commodities -> {OUTPUT_FILE}")
