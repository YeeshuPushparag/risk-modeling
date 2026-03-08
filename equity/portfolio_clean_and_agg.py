import os
import re
import requests
from io import StringIO
from datetime import datetime

import pandas as pd

# ======================
# BASE PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_VANGUARD = os.path.join(BASE_DIR, "data", "vanguard_13f")
RAW_BLACKROCK = os.path.join(BASE_DIR, "data", "blackrock_13f")

FINAL_OUTPUT = os.path.join(BASE_DIR, "data", "final_13f.csv")
SP500_OUTPUT = os.path.join(BASE_DIR, "data", "sp500_filtered.csv")
TICKERS_OUTPUT = os.path.join(BASE_DIR, "data", "all_tickers.csv")
TICKERS_MAXDATE_OUTPUT = os.path.join(BASE_DIR, "data", "max_date_tickers.csv")


# ======================
# HELPERS
# ======================
def get_quarter_end_date(q, year):
    q = q.upper()
    return {
        "Q1": datetime(year, 3, 31),
        "Q2": datetime(year, 6, 30),
        "Q3": datetime(year, 9, 30),
        "Q4": datetime(year, 12, 31),
    }[q]


# ======================
# STEP 1: LOAD + TAG FILES (PANDAS)
# ======================
def load_13f(folder, company):
    pattern = rf"{company}.* (Q[1-4]) (\d{{4}})"

    dfs = []

    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        if company not in fname:
            continue

        m = re.search(pattern, fname)
        if not m:
            print("Skipped:", fname)
            continue

        q, year = m.group(1), int(m.group(2))
        q_end = get_quarter_end_date(q, year)

        path = os.path.join(folder, fname)
        df = pd.read_csv(path)

        df["date"] = q_end.date()
        df["asset_manager"] = company

        print("Loaded:", fname, "->", q_end.date())
        dfs.append(df)

    return dfs


# ======================
# STEP 2: MERGE V + B (PANDAS CONCAT)
# ======================
def merge_all(v_list, b_list):
    all_dfs = v_list + b_list

    if not all_dfs:
        raise ValueError("No 13F files found.")

    merged = pd.concat(all_dfs, ignore_index=True)
    print("Merged total rows:", len(merged))
    return merged


# ======================
# STEP 3: CLEAN DATA
# ======================
def clean_data(df):

    # Drop column
    if "Principal" in df.columns:
        df = df.drop(columns=["Principal"])

    rename_map = {
        "Sym": "ticker",
        "Issuer Name": "issuer_name",
        "Cl": "class",
        "Value ($000)": "total_value",
        "%": "total_percent",
        "Option Type": "option_type",
        "Other Manager": "other_manager"
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Clean option_type
    df["option_type"] = df["option_type"].fillna("").astype(str)
    df["option_type"] = df["option_type"].replace("", "Stock")

    # Clean other_manager
    df["other_manager"] = df["other_manager"].fillna("").astype(str)
    df["other_manager"] = df["other_manager"].replace("", "0")

    # Drop rows with missing ticker or shares
    df = df[df["ticker"].notna()]
    df = df[df["Shares"].notna()]

    return df


# ======================
# STEP 4: AGGREGATE (PANDAS GROUPBY)
# ======================
def aggregate(df):

    # Convert numeric columns
    numeric = ["total_value", "Shares", "Sole", "Shared", "other_manager", "None"]
    for c in numeric:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

    if "total_percent" in df.columns:
        df["total_percent"] = (
            df["total_percent"].astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    # Groupby same as Spark
    agg = df.groupby(["ticker", "date", "asset_manager"], as_index=False).agg({
        "issuer_name": "first",
        "class": lambda x: "MIXED" if x.nunique() > 1 else x.iloc[0],
        "CUSIP": "first",
        "total_value": "sum",
        "total_percent": "first",
        "Shares": "sum",
        "option_type": lambda x: "MIXED" if x.nunique() > 1 else x.iloc[0],
        "Discretion": "first",
        "other_manager": "first",
        "Sole": "sum",
        "Shared": "sum",
        "None": "sum",
    })

    return agg


# ======================
# STEP 5: FILTER S&P 500 USING WIKIPEDIA
# ======================
def filter_sp500(df):
    """
    Extract unique tickers from df and keep only those 
    that are in the official S&P500 list and whose last date >= Q2 2024.
    """
    # --- 0. Remove GOOG ticker ---
    df = df[df["ticker"].str.upper() != "GOOG"]

    # --- 1. Compute last date per ticker and keep only those >= 2024-Q2 ---
    df["date"] = pd.to_datetime(df["date"])
    last_dates = df.groupby("ticker")["date"].max().reset_index()
    cutoff_date = pd.Timestamp("2024-06-30")
    symbols = last_dates.loc[last_dates["date"] >= cutoff_date, "ticker"].astype(str).str.strip().tolist()

    if len(symbols) == 0:
        print("No tickers have data beyond Q2 2024.")
        return []

    # --- 2. Load S&P 500 from Wikipedia ---
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    tables = pd.read_html(StringIO(html))

    sp_table = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "symbol" in cols:
            t.columns = cols
            sp_table = t
            break

    if sp_table is None:
        raise ValueError("S&P500 table not found.")

    sp500 = (
        sp_table["symbol"]
        .astype(str)
        .str.replace(".", "-", regex=False)   # Yahoo adjustment
        .str.upper()
        .tolist()
    )

    # --- 3. Filter tickers by S&P500 membership ---
    filtered = sorted([s for s in symbols if s.upper() in sp500])

    # --- 4. Save as CSV ---
    pd.DataFrame({"ticker": filtered}).to_csv(SP500_OUTPUT, index=False)

    print(f"S&P500 tickers with data >= Q2 2024: {len(filtered)}")
    print("Saved:", SP500_OUTPUT)

    return filtered



# ======================
# MAIN PIPELINE
# ======================
def run_pipeline():
    print("START PIPELINE")

    v_list = load_13f(RAW_VANGUARD, "Vanguard")
    b_list = load_13f(RAW_BLACKROCK, "BlackRock")

    merged = merge_all(v_list, b_list)
    cleaned = clean_data(merged)
    aggregated = aggregate(cleaned)

    aggregated.to_csv(FINAL_OUTPUT, index=False)
    print("Saved aggregated CSV ->", FINAL_OUTPUT)

    # --- Save ALL unique tickers ---
    all_tickers = aggregated['ticker'].unique()
    pd.DataFrame(all_tickers, columns=["ticker"]).to_csv(TICKERS_OUTPUT, index=False)

    print(f"All unique tickers: {len(all_tickers)}")
    print(f"Saved ALL tickers -> {TICKERS_OUTPUT}")


    # --- Save ONLY tickers with MAX DATE ---
    aggregated['date'] = pd.to_datetime(aggregated['date'])
    max_date = aggregated['date'].max()

    latest_rows = aggregated[aggregated['date'] == max_date]
    latest_tickers = latest_rows['ticker'].unique()

    # Output path for latest tickers
    TICKERS_MAXDATE_OUTPUT = os.path.join(BASE_DIR, "data", "tickers_max_date.csv")

    pd.DataFrame(latest_tickers, columns=["ticker"]).to_csv(TICKERS_MAXDATE_OUTPUT, index=False)

    print(f"Tickers with max date ({max_date.date()}): {len(latest_tickers)}")
    print(f"Saved MAX-DATE tickers -> {TICKERS_MAXDATE_OUTPUT}")


    filter_sp500(aggregated)

    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    run_pipeline()
