"""
Unified fundamentals pipeline with metadata imputation.
Fetches all Yahoo Finance data once per ticker and ensures no missing beta/dividendYield.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# ===========================
# CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_TICKERS = os.path.join(BASE_DIR, "data", "sp500_filtered.csv")

OUTPUT_CSV = os.path.join(BASE_DIR, "data", "corporate_fundamentals1.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===========================
# UTILITIES
# ===========================
def normalize_quarter(date):
    """Normalize any fiscal date to the nearest quarter-end."""
    if pd.isna(date):
        return None
    d = pd.to_datetime(date)
    if d.month in [1, 2, 3]:
        return f"{d.year}-03-31"
    elif d.month in [4, 5, 6]:
        return f"{d.year}-06-30"
    elif d.month in [7, 8, 9]:
        return f"{d.year}-09-30"
    else:
        return f"{d.year}-12-31"

# ===========================
# FETCH LOOP
# ===========================
tickers = pd.read_csv(INPUT_TICKERS)["ticker"].dropna().unique().tolist()
print(f"Loaded {len(tickers)} tickers from {INPUT_TICKERS}")
records = []

for t in tqdm(tickers, desc="Fetching fundamentals", ncols=100):
    try:
        tk = yf.Ticker(t)
        info = tk.info
        bs = tk.quarterly_balance_sheet.T
        is_ = tk.quarterly_financials.T

        if bs.empty or is_.empty:
            continue

        for date in bs.index:
            rec = {
                "ticker": t,
                "date": normalize_quarter(date),
                # --- Balance Sheet ---
                "totalAssets": bs.get("Total Assets", {}).get(date),
                "totalDebt": (
                    bs.get("Total Debt", {}).get(date)
                    or bs.get("Total Liabilities Net Minority Interest", {}).get(date)
                    or bs.get("Long Term Debt", {}).get(date)
                ),
                "shortTermDebt": (
                    bs.get("Short Long Term Debt", {}).get(date)
                    or bs.get("Short Term Debt", {}).get(date)
                    or bs.get("Current Debt", {}).get(date)
                ),
                "longTermDebt": (
                    bs.get("Long Term Debt", {}).get(date)
                    or bs.get("Noncurrent Debt", {}).get(date)
                ),
                # --- Income Statement ---
                "revenue": is_.get("Total Revenue", {}).get(date),
                "netIncome": is_.get("Net Income", {}).get(date),
                "ebitda": is_.get("EBITDA", {}).get(date),
                # --- Market Metadata ---
                "marketCap": info.get("marketCap", np.nan),
                "beta": info.get("beta", np.nan),
                "dividendYield": info.get("dividendYield", np.nan),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }
            records.append(rec)
    except Exception:
        continue

df = pd.DataFrame(records)
df.dropna(subset=["ticker", "date"], inplace=True)
df = df.groupby(["ticker", "date"], as_index=False).first()
df["date"] = pd.to_datetime(df["date"])

# Restrict to valid quarters only
quarters = pd.date_range("2023-12-31", "2025-09-30", freq="QE")
df = df[df["date"].isin(quarters)].reset_index(drop=True)

# ===========================
# CLEANING & BASIC IMPUTATION
# ===========================
numeric_cols = [
    "totalAssets", "totalDebt", "shortTermDebt", "longTermDebt",
    "revenue", "netIncome", "ebitda", "marketCap", "beta", "dividendYield"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Debt cleaning
df["shortTermDebt"] = df["shortTermDebt"].fillna(0)
mask_long = df["longTermDebt"].isna() & df["totalDebt"].notna()
df.loc[mask_long, "longTermDebt"] = (df.loc[mask_long, "totalDebt"] - df.loc[mask_long, "shortTermDebt"]).clip(lower=0)
df["longTermDebt"] = df["longTermDebt"].fillna(0)
df["totalDebt"] = df["totalDebt"].fillna(df["shortTermDebt"] + df["longTermDebt"])

# EBITDA imputation (per-ticker median ratio)
mask_ebitda = df["ebitda"].isna()
if mask_ebitda.any():
    df["ebitda_ratio"] = df["ebitda"] / df["revenue"]
    med_ratio = df.groupby("ticker")["ebitda_ratio"].median()
    def fill_ebitda(row):
        if not np.isnan(row["ebitda"]):
            return row["ebitda"]
        ratio = med_ratio.get(row["ticker"], np.nan)
        return row["revenue"] * ratio if pd.notna(ratio) else np.nan
    df["ebitda"] = df.apply(fill_ebitda, axis=1)
    df.drop(columns=["ebitda_ratio"], inplace=True)

mask_ebitda2 = df["ebitda"].isna()
if mask_ebitda2.any():
    global_ratio = (df["ebitda"] / df["revenue"]).median(skipna=True)
    df.loc[mask_ebitda2, "ebitda"] = df.loc[mask_ebitda2, "revenue"] * global_ratio

# ===========================
# IMPUTE MISSING METADATA
# ===========================
# Replace missing sectors/industries with 'Unknown'
df["sector"] = df["sector"].fillna("Unknown")
df["industry"] = df["industry"].fillna("Unknown")

# --- Dividend Yield ---
div_sector_median = df.groupby("sector")["dividendYield"].median()
def fill_div_yield(row):
    if not np.isnan(row["dividendYield"]):
        return row["dividendYield"]
    sec_median = div_sector_median.get(row["sector"], np.nan)
    return sec_median if not np.isnan(sec_median) else df["dividendYield"].median(skipna=True)
df["dividendYield"] = df.apply(fill_div_yield, axis=1)

# --- Beta ---
beta_industry_median = df.groupby("industry")["beta"].median()
def fill_beta(row):
    if not np.isnan(row["beta"]):
        return row["beta"]
    ind_median = beta_industry_median.get(row["industry"], np.nan)
    return ind_median if not np.isnan(ind_median) else df["beta"].median(skipna=True)
df["beta"] = df.apply(fill_beta, axis=1)

# ===========================
# DERIVED RATIOS
# ===========================
df["debt_to_assets"] = df["totalDebt"] / df["totalAssets"]
df["debt_to_ebitda"] = df["totalDebt"] / df["ebitda"].replace(0, np.nan)
df["ebitda_margin"] = df["ebitda"] / df["revenue"]

# ===========================
# FINALIZE
# ===========================
df = df.dropna(subset=["revenue", "netIncome", "totalAssets", "totalDebt"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n Saved {len(df)} records across {df['ticker'].nunique()} tickers -> {OUTPUT_CSV}")

missing_div = df["dividendYield"].isna().mean() * 100
missing_beta = df["beta"].isna().mean() * 100
print(f"Missing DividendYield: {missing_div:.2f}% | Missing Beta: {missing_beta:.2f}%")
