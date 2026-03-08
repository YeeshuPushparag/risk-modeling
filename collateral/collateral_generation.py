import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

HAIRCUT = {
    "Cash": 0.00, "Treasury": 0.02, "CorporateBond": 0.10,
    "Equity": 0.15, "CommodityETF": 0.12
}
INITIAL_MARGIN = {
    "FX": 0.05, "Commodity": 0.08, "Bond": 0.04, "Equity": 0.12
}
COLLATERAL_MENU = {
    "FX": [("Cash", 0.70), ("Treasury", 0.25), ("CorporateBond", 0.05)],
    "Commodity": [("Treasury", 0.60), ("Cash", 0.25), ("CommodityETF", 0.15)],
    "Bond": [("Treasury", 0.80), ("Cash", 0.15), ("CorporateBond", 0.05)],
    "Equity": [("Cash", 0.50), ("Treasury", 0.30), ("Equity", 0.20)],
}
AGREEMENT_CHOICES = [
    ("CSA", "US", 0.45),
    ("CSA", "UK", 0.25),
    ("CSA", "EU", 0.20),
    ("GMRA", "UK", 0.05),
    ("GMSLA", "US", 0.05),
]

# --- HELPERS ---
def parse_dates_safe(series: pd.Series) -> pd.Series:
    """Vectorized date parser keeping all rows."""
    dates = pd.to_datetime(series, errors='coerce')
    if dates.notna().sum() < len(series) * 0.8:
        dates_alt = pd.to_datetime(series, dayfirst=True, errors='coerce')
        dates = dates_alt if dates_alt.notna().sum() > dates.notna().sum() else dates
    return dates

def fast_counterparty(series: pd.Series) -> pd.Series:
    """Vectorized counterparty assignment."""
    buckets = np.array(["CP_A", "CP_B", "CP_C", "CP_D", "CP_E"])
    hashes = series.astype(str).apply(lambda x: int(hashlib.sha256(x.encode()).hexdigest(), 16))
    indices = (hashes % len(buckets)).astype(int)
    return pd.Series(buckets[indices], index=series.index)

def choose_weighted_vectorized(opts, size: int) -> np.ndarray:
    """Vectorized weighted choice from list of (name, prob)."""
    names, probs = zip(*opts)
    probs = np.array(probs) / np.sum(probs)
    return np.random.choice(names, size=size, p=probs)

def ensure_cols(df, cols):
    """Ensure required columns exist in dataframe."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# --- BUILD COLLATERAL BLOCK ---
def build_collateral_block(df, asset_class, sym_col, exposure_col, sector_col="sector", industry_col="industry"):
    if df is None or df.empty:
        return None

    df = ensure_cols(df, [sym_col, exposure_col, sector_col, industry_col, "date"])
    df["date"] = parse_dates_safe(df["date"])

    out = pd.DataFrame()
    out["date"] = df["date"]
    out["asset_class"] = asset_class
    out["ticker"] = df[sym_col].astype(str)
    out["sector"] = df[sector_col].fillna("Unknown")
    out["industry"] = df[industry_col].fillna("Unknown")
    out["counterparty"] = fast_counterparty(df[sym_col])
    out["exposure_before_collateral"] = df[exposure_col].abs().astype(float)

    # Agreement type
    choices, probs = zip(*[((a, j), p) for a, j, p in AGREEMENT_CHOICES])
    idx = np.random.choice(len(choices), len(out), p=np.array(probs)/np.sum(probs))
    agreements, jurisdictions = zip(*[choices[i] for i in idx])
    out["agreement_type"], out["jurisdiction"] = agreements, jurisdictions

    # Collateral
    out["collateral_type"] = choose_weighted_vectorized(COLLATERAL_MENU[asset_class], len(out))
    out["haircut"] = out["collateral_type"].map(HAIRCUT)
    out["initial_margin_rate"] = INITIAL_MARGIN[asset_class]
    out["required_collateral"] = out["exposure_before_collateral"] * out["initial_margin_rate"]
    out["collateral_value"] = 0.9 * out["required_collateral"]
    out["effective_collateral"] = out["collateral_value"] * (1 - out["haircut"])
    out["net_exposure"] = (out["exposure_before_collateral"] - out["effective_collateral"]).clip(lower=0)
    out["collateral_ratio"] = np.where(
        out["exposure_before_collateral"] > 0,
        out["effective_collateral"] / out["exposure_before_collateral"],
        0
    )
    out["margin_call_flag"] = (out["net_exposure"] > 0).astype(int)
    out["margin_call_amount"] = out["net_exposure"]
    out["reuse_flag"] = np.random.binomial(1, 0.35, len(out))
    out["reused_value"] = out["collateral_value"] * out["reuse_flag"] * 0.8
    out["funding_cost"] = out["collateral_value"] * 0.01 * (out["haircut"] + out["initial_margin_rate"])
    out["liquidity_score"] = 1 - (out["haircut"] + out["initial_margin_rate"])

    # Trade & Collateral IDs
    out["trade_id"] = asset_class[:3].upper() + "_" + out["ticker"] + "_" + out["date"].dt.strftime("%Y%m%d")
    concat = out["trade_id"] + "|" + out["counterparty"] + "|" + out["date"].astype(str) + "|" + out.index.astype(str)
    out["collateral_id"] = pd.Series([hashlib.md5(x.encode("utf-8")).hexdigest() for x in concat], index=out.index)

    return out

# --- MAIN ---
print("Starting collateral CSV data generation...")

input_files = {
    "FX": ("fx.csv", "ticker", "exposure_amount"),
    "Commodity": ("commodities.csv", "ticker", "exposure_amount"),
    "Bond": ("bonds.csv", "ticker", "bond_price"),
    "Equity": ("equity.csv", "ticker", "mtm_value")
}

rows = []
for asset_class, (file_name, sym_col, exp_col) in input_files.items():
    file_path = Path(DATA_DIR) / "final" / file_name
    if file_path.exists():
        df = pd.read_csv(file_path, low_memory=False)
        rows.append(build_collateral_block(df, asset_class, sym_col, exp_col))
    else:
        print(f"{file_name} not found — skipping {asset_class}.")

if not rows:
    raise RuntimeError("No data found for collateral generation.")

collateral_full = pd.concat(rows, ignore_index=True).sort_values(
    ["date", "counterparty", "asset_class", "ticker"]
)

output_path = Path(DATA_DIR) / "collateral_daily_detailed.csv"
collateral_full.to_csv(output_path, index=False)

print(f"Detailed collateral dataset saved: {len(collateral_full):,} rows")
print(f"Output file: {output_path}")
print(f"Date range: {collateral_full['date'].min()} to {collateral_full['date'].max()}")
print("Asset class distribution:")
print(collateral_full['asset_class'].value_counts())
