import pandas as pd
import numpy as np
from dateutil import parser
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT = os.path.join(DATA_DIR, "derivatives_positions.csv")
MACRO = os.path.join(DATA_DIR, "macro_data.csv")
OUTPUT = os.path.join(DATA_DIR, "final", "derivatives.csv")

def force_date(x):
    try: return parser.parse(str(x), fuzzy=True, dayfirst=True)
    except Exception:
        try: return parser.parse(str(x), fuzzy=True)
        except Exception: return pd.NaT

start_time = time.time()
df = pd.read_csv(INPUT, low_memory=False)

# === Fill Greeks ===
for c in ["delta","gamma","vega"]:
    if c not in df.columns:
        df[c] = np.nan

mask_all = df[["delta","gamma","vega"]].isna().all(axis=1)
df.loc[mask_all, ["delta","gamma","vega"]] = [1.0, 0.0, 0.0]
for c in ["delta","gamma","vega"]:
    df[c] = df[c].fillna(df[c].median(skipna=True)).clip(lower=0)

# === Enrich Risk ===
df["delta_equivalent_exposure"] = df["delta"] * df["notional"]
np.random.seed(42)
rand_years = np.random.choice(np.arange(0.5, 5.1, 0.5), size=len(df))
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["maturity_date"] = df["date"] + pd.to_timedelta((rand_years * 365).round(), unit="D")
df["tenor_years"] = rand_years

vol = df.groupby("asset_class")["underlying_price"].transform("std").fillna(0)
df["pnl"] = df["delta_equivalent_exposure"] * 0.001 * (vol / (vol.max() or 1))

macro = pd.read_csv(MACRO)
df["date"] = pd.to_datetime(df["date"])
macro["date"] = pd.to_datetime(macro["date"])
df["month_year"] = df["date"].dt.to_period("M").astype(str)
macro["month_year"] = macro["date"].dt.to_period("M").astype(str)

merged = df.merge(
    macro.drop(columns=["date"]),
    on="month_year",
    how="left"
).drop(columns=["month_year"])



merged.to_csv(OUTPUT, index=False)
elapsed = time.time() - start_time
print(f"Final derivatives saved -> final/derivates | Rows: {len(df):,} | Elapsed: {elapsed:,.2f}s")
