import os
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BOND_INPUT = os.path.join(DATA_DIR, "final", "bonds.csv")
COMM_INPUT = os.path.join(DATA_DIR, "final", "commodities.csv")
EQUITY_INPUT = os.path.join(DATA_DIR, "final", "equity.csv")
FX_INPUT = os.path.join(DATA_DIR, "final", "fx.csv")
OUT_PATH = os.path.join(DATA_DIR, "derivatives_positions.csv")

INITIAL_MARGIN = {"FX":0.05,"Commodity":0.08,"Bond":0.04,"Equity":0.10}
HAIRCUT = {"Cash":0.00,"Treasury":0.02,"CorporateBond":0.10,"Equity":0.15}
COLLATERAL_TYPE = {"FX":"Cash","Commodity":"Treasury","Bond":"Treasury","Equity":"Equity"}

# --- BETTER DATE PARSING ---
def parse_dates_safe(series):
    """Safe date parsing that tries multiple approaches"""
    # First try without dayfirst (for standard YYYY-MM-DD format)
    dates1 = pd.to_datetime(series, errors='coerce')
    
    # Count how many were successfully parsed
    success_count1 = dates1.notna().sum()
    
    # If we lost many dates, try with dayfirst=True
    if success_count1 < len(series) * 0.8:  # If we lost more than 20%
        dates2 = pd.to_datetime(series, dayfirst=True, errors='coerce')
        success_count2 = dates2.notna().sum()
        
        # Use whichever method parsed more dates
        if success_count2 > success_count1:
            print(f"  Using dayfirst=True (parsed {success_count2:,} vs {success_count1:,} dates)")
            return dates2
        else:
            print(f"  Using default parsing (parsed {success_count1:,} vs {success_count2:,} dates)")
            return dates1
    else:
        print(f"  Using default parsing (parsed {success_count1:,} dates)")
        return dates1

def fast_counterparty(series):
    buckets = np.array(["CP_A","CP_B","CP_C","CP_D","CP_E"])
    return series.astype(str).apply(
        lambda s: buckets[int(hashlib.sha256(s.encode()).hexdigest(),16)%len(buckets)]
    )

def generate_trade_id(row):
    """Generate unique trade_id with cleaned commodity_sym and asset_manager."""
    parts = [row['derivative_type'], row['ticker'], row['date'].strftime("%Y%m%d")]
    
    if pd.notna(row.get('commodity_sym')):
        clean_sym = str(row['commodity_sym']).replace('=', '_')
        parts.append(clean_sym)
    
    if pd.notna(row.get('asset_manager')):
        clean_mgr = str(row['asset_manager']).split()[0]
        parts.append(clean_mgr)
    
    return "_".join(parts)

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def section(title):
    print(f"\n{'='*80}\n[{title.upper()}]\n{'='*80}")

# --- MAIN ---
start_time = time.time()
frames = []

# ---------------- FX ----------------
section("FX LOAD")
if Path(FX_INPUT).exists():
    fx = pd.read_csv(FX_INPUT)
    print(f"FX raw data: {len(fx):,} rows")
    print(f"Sample dates: {fx['date'].head(3).tolist()}")
    
    # Safe date parsing
    fx["date"] = parse_dates_safe(fx["date"])
    initial_count = len(fx)
    fx = fx[fx["date"].notna()]
    print(f"FX after date cleaning: {len(fx):,} rows (removed {initial_count - len(fx):,} NaT dates)")
    
    fx = ensure_cols(fx, ["ticker","sector","industry","fx_rate","exposure_amount"])
    
    fx_clean = pd.DataFrame({
        "date": fx["date"], "asset_class": "FX", "derivative_type": "FXFwd",
        "ticker": fx["ticker"], "sector": fx["sector"], "industry": fx["industry"],
        "underlying_price": fx["fx_rate"], "notional": fx["exposure_amount"].abs(),
        "delta": 1.0, "gamma": 0.0, "vega": 0.0,
        "initial_margin_rate": INITIAL_MARGIN["FX"],
        "collateral_type": COLLATERAL_TYPE["FX"], "haircut": HAIRCUT["Cash"],
        "asset_manager": None,  
        "commodity_sym": None,   
    })
    
    fx_clean = fx_clean[fx_clean["date"].notna()]
    frames.append(fx_clean)
    print(f"FX final: {len(fx_clean):,} rows")
else:
    print("FX file not found — skipping.")

# ---------------- COMMODITIES ----------------
section("COMMODITIES LOAD")
if Path(COMM_INPUT).exists():
    cmd = pd.read_csv(COMM_INPUT)
    print(f"Commodities raw data: {len(cmd):,} rows")
    print(f"Sample dates: {cmd['date'].head(3).tolist()}")
    
    cmd["date"] = parse_dates_safe(cmd["date"])
    initial_count = len(cmd)
    cmd = cmd[cmd["date"].notna()]
    print(f"Commodities after date cleaning: {len(cmd):,} rows (removed {initial_count - len(cmd):,} NaT dates)")
    
    cmd = ensure_cols(cmd, ["ticker","sector","industry","close","exposure_amount", "asset_manager", "commodity"])
    
    cmd_clean = pd.DataFrame({
        "date": cmd["date"], "asset_class": "Commodity", "derivative_type": "Futures",
        "ticker": cmd["ticker"], "sector": cmd["sector"], "industry": cmd["industry"],
        "underlying_price": cmd["close"], "notional": cmd["exposure_amount"].abs(),
        "delta": 1.0, "gamma": 0.0, "vega": 0.0,
        "initial_margin_rate": INITIAL_MARGIN["Commodity"],
        "collateral_type": COLLATERAL_TYPE["Commodity"], "haircut": HAIRCUT["Treasury"],
        "asset_manager": cmd["asset_manager"],  
        "commodity_sym": cmd["commodity"],   
    })
    
    cmd_clean = cmd_clean[cmd_clean["date"].notna()]
    frames.append(cmd_clean)
    print(f"Commodities final: {len(cmd_clean):,} rows")
else:
    print("Commodities file not found — skipping.")

# ---------------- BONDS ----------------
section("BONDS LOAD")
if Path(BOND_INPUT).exists():
    bnd = pd.read_csv(BOND_INPUT)
    print(f"Bonds raw data: {len(bnd):,} rows")
    print(f"Sample dates: {bnd['date'].head(3).tolist()}")
    
    bnd["date"] = parse_dates_safe(bnd["date"])
    initial_count = len(bnd)
    bnd = bnd[bnd["date"].notna()]
    print(f"Bonds after date cleaning: {len(bnd):,} rows (removed {initial_count - len(bnd):,} NaT dates)")
    
    bnd = ensure_cols(bnd, ["ticker","sector","industry","yield_to_maturity","maturity_years"])
    
    bnd_clean = pd.DataFrame({
        "date": bnd["date"], "asset_class": "Bond", "derivative_type": "IRS",
        "ticker": bnd["ticker"], "sector": bnd["sector"], "industry": bnd["industry"],
        "underlying_price": bnd["yield_to_maturity"],
        "notional": (bnd["maturity_years"].abs()*1_000_000).fillna(1_000_000),
        "delta": np.nan, "gamma": np.nan, "vega": np.nan,
        "initial_margin_rate": INITIAL_MARGIN["Bond"],
        "collateral_type": COLLATERAL_TYPE["Bond"], "haircut": HAIRCUT["Treasury"],
        "asset_manager": None,  
        "commodity_sym": None,   
    })
    
    bnd_clean = bnd_clean[bnd_clean["date"].notna()]
    frames.append(bnd_clean)
    print(f"Bonds final: {len(bnd_clean):,} rows")
else:
    print("Bonds file not found — skipping.")

# ---------------- EQUITY ----------------
section("EQUITY LOAD")
if Path(EQUITY_INPUT).exists():
    eq = pd.read_csv(EQUITY_INPUT)
    print(f"Equity raw data: {len(eq):,} rows")
    print(f"Sample dates: {eq['date'].head(3).tolist()}")
    
    eq["date"] = parse_dates_safe(eq["date"])
    initial_count = len(eq)
    eq = eq[eq["date"].notna()]
    print(f"Equity after date cleaning: {len(eq):,} rows (removed {initial_count - len(eq):,} NaT dates)")
    
    eq = ensure_cols(eq, ["ticker","sector","industry","close","mtm_value", "asset_manager"])
    
    eq_clean = pd.DataFrame({
        "date": eq["date"], "asset_class": "Equity", "derivative_type": "EqFwd",
        "ticker": eq["ticker"], "sector": eq["sector"], "industry": eq["industry"],
        "underlying_price": eq["close"], "notional": eq["mtm_value"].abs(),
        "delta": 1.0, "gamma": 0.0, "vega": 0.0,
        "initial_margin_rate": INITIAL_MARGIN["Equity"],
        "collateral_type": COLLATERAL_TYPE["Equity"], "haircut": HAIRCUT["Equity"],
        "asset_manager": eq["asset_manager"],  
        "commodity_sym": None,   
    })
    
    eq_clean = eq_clean[eq_clean["date"].notna()]
    frames.append(eq_clean)
    print(f"Equity final: {len(eq_clean):,} rows")
else:
    print("Equity file not found — skipping.")

# --- Combine ---
if not frames:
    raise RuntimeError("No derivatives data found.")

print(f"\nCombining {len(frames)} datasets...")
df = pd.concat(frames, ignore_index=True)

# Remove any final NaT dates
initial_combined = len(df)
df = df[df["date"].notna()]
print(f"After final date cleaning: {len(df):,} rows (removed {initial_combined - len(df):,} NaT dates)")

# Counterparty and trade_id
df["counterparty"] = fast_counterparty(df["ticker"])
df["trade_id"] = df.apply(generate_trade_id, axis=1)
# --- Risk Calculation ---
df["exposure_before_collateral"] = df["notional"].abs()
df["required_collateral"] = df["exposure_before_collateral"] * df["initial_margin_rate"]
df["collateral_value"] = 0.9 * df["required_collateral"]
df["effective_collateral"] = df["collateral_value"] * (1 - df["haircut"])
df["net_exposure"] = (df["exposure_before_collateral"] - df["effective_collateral"]).clip(lower=0)
df["collateral_ratio"] = np.where(
    df["exposure_before_collateral"] > 0,
    df["effective_collateral"] / df["exposure_before_collateral"],
    0
)
df["margin_call_flag"] = (df["net_exposure"] > 0).astype(int)
df["margin_call_amount"] = df["net_exposure"]
df = df.drop(columns=['commodity_sym', 'asset_manager'], errors='ignore')
# --- Output ---
cols = [
    "trade_id","date","counterparty","asset_class","derivative_type","ticker","sector","industry",
    "underlying_price","notional","delta","gamma","vega",
    "exposure_before_collateral","initial_margin_rate","collateral_type","haircut",
    "required_collateral","collateral_value","effective_collateral","net_exposure",
    "collateral_ratio","margin_call_flag","margin_call_amount"
]
df = df[cols].sort_values(["date","asset_class","ticker"])
df.to_csv(OUT_PATH, index=False)

elapsed = time.time()-start_time
print(f"\n Derivatives positions saved -> {OUT_PATH}")
print(f" Rows: {len(df):,} | Time: {elapsed:.2f}s")
print(f" Date range: {df['date'].min()} to {df['date'].max()}")
print(f" Year distribution: {df['date'].dt.year.value_counts().to_dict()}")
print(f" Asset classes: {df['asset_class'].value_counts().to_dict()}")