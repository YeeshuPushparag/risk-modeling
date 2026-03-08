import os
import pandas as pd

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FINAL_DIR = os.path.join(DATA_DIR, "final")

# Input files
INPUT_COMMOD = os.path.join(DATA_DIR, "commodities_daily.csv")
SYM = os.path.join(DATA_DIR, "unique_sym_sector_industry.csv")
MACRO = os.path.join(DATA_DIR, "macro_data.csv")
TICKER_DETAIL = os.path.join(DATA_DIR, "ticker_details.csv")  # <-- NEW FILE WITH MTM

# Output file
OUTPUT = os.path.join(FINAL_DIR, "commodities1.csv")

print("[1] Loading data...")
companies = pd.read_csv(SYM)
commod_base = pd.read_csv(INPUT_COMMOD)
macro = pd.read_csv(MACRO, parse_dates=["date"])
ticker_detail = pd.read_csv(TICKER_DETAIL)      # <-- load mtm values
ticker_detail["date"] = pd.to_datetime(ticker_detail["date"])

# Clean commodity data
commod_base["date"] = pd.to_datetime(commod_base["date"], errors="coerce")
commod_base.rename(columns={"commodity_symbol": "commodity"}, inplace=True)

# === SECTOR → COMMODITY WEIGHTS ===
sector_to_commodities = {
    "Energy": {"CL=F": 0.7, "NG=F": 0.3},
    "Basic Materials": {"GC=F": 0.3, "SI=F": 0.3, "ZC=F": 0.4},
    "Industrials": {"CL=F": 0.4, "GC=F": 0.6},
    "Consumer Defensive": {"ZC=F": 0.6, "GC=F": 0.4},
    "Utilities": {"NG=F": 0.8, "CL=F": 0.2},
    "Technology": {"GC=F": 0.7, "SI=F": 0.3},
    "Healthcare": {"SI=F": 0.4, "GC=F": 0.6},
    "Financial Services": {"GC=F": 0.8, "CL=F": 0.2},
    "Real Estate": {"GC=F": 0.5, "ZC=F": 0.5},
    "Communication Services": {"GC=F": 0.5, "CL=F": 0.5},
    "Consumer Cyclical": {"CL=F": 0.6, "GC=F": 0.4},
}

print("[2] Building exposure base (ticker x commodity)...")
rows = []
for _, r in companies.iterrows():
    mapping = sector_to_commodities.get(r["sector"])
    if not mapping:
        continue
    for comm, weight in mapping.items():
        rows.append({
            "ticker": r["ticker"],
            "sector": r["sector"],
            "industry": r["industry"],
            "commodity": comm,
            "sensitivity": weight
        })
exp = pd.DataFrame(rows)

print(f"Exposure base: {len(exp):,} rows")

# === MERGE WITH COMMODITY DATA ===
print("[3] Merging with commodity daily data...")
seg = exp.merge(commod_base, on="commodity", how="left", validate="m:m")

# === HEDGE RATIO ONLY (no PNL yet) ===
seg["vol_20d"] = pd.to_numeric(seg["vol_20d"], errors="coerce")
seg["daily_return"] = pd.to_numeric(seg["daily_return"], errors="coerce").fillna(0.0)
seg["hedge_ratio"] = (0.2 + 0.6 * seg["vol_20d"].rank(pct=True)).clip(0, 1)

# === MERGE WITH MACRO ===
print("[4] Merging with macro data...")
seg["date"] = pd.to_datetime(seg["date"])
macro["date"] = pd.to_datetime(macro["date"])

seg["mm_yy"] = seg["date"].dt.strftime("%m-%y")
macro["mm_yy"] = macro["date"].dt.strftime("%m-%y")

macro_for_merge = macro.drop(columns=["date"])

seg = seg.merge(macro_for_merge, on="mm_yy", how="left").drop(columns=["mm_yy"])

# === FILTER DATES ===
seg = seg[seg["date"] >= "2024-04-01"]

print(f"After filtering: {len(seg):,} rows")

# === NOW MERGE DAILY MTM VALUE (IMPORTANT!) ===
print("[5] Merging ticker mtm values (per-day)...")
seg = seg.merge(
    ticker_detail[["ticker", "date", "mtm_value", "asset_manager"]],
    on=["ticker", "date"],
    how="inner"
)

print(f"After adding mtm_value: {len(seg):,} rows")

# === NOW CALCULATE TRUE EXPOSURE & PNL ===
print("[6] Calculating exposure and pnl...")

seg["mtm_value"] = pd.to_numeric(seg["mtm_value"], errors="coerce").fillna(0)

seg["exposure_amount"] = seg["sensitivity"] * seg["mtm_value"]
seg["commodity_pnl"] = (
    seg["exposure_amount"] *
    seg["daily_return"] *
    (1 - seg["hedge_ratio"])
)

# === FINAL OUTPUT ===
cols = [
    "ticker", "asset_manager", "sector", "industry", "commodity", "date",
    "open", "high", "low", "close", "volume",
    "daily_return", "log_return", "vol_20d", 
    "sensitivity", "hedge_ratio", "mtm_value",
    "exposure_amount", "commodity_pnl",
    "VaR_95", "VaR_99", "gdp", "unrate", "cpi", "fedfunds"
]

available_cols = [c for c in cols if c in seg.columns]
final = seg[available_cols].sort_values(["commodity", "date", "ticker"])

# Save
final.to_csv(OUTPUT, index=False)
print(f"\n Final commodities saved -> {OUTPUT} | Rows: {len(final):,}")
