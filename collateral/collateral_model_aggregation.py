import os
import pandas as pd

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def create_collateral_model():
    """Create aggregated collateral model dataset"""
    collateral_full = pd.read_csv(os.path.join(DATA_DIR, "collateral_daily_detailed.csv"), parse_dates=["date"])
    
    collateral_model = (
        collateral_full.groupby(["date", "asset_class", "ticker", "counterparty"], as_index=False)
        .agg({
            "exposure_before_collateral": "sum",
            "required_collateral": "sum",
            "collateral_value": "sum",
            "effective_collateral": "sum",
            "net_exposure": "sum",
            "margin_call_amount": "sum",
            "funding_cost": "sum",
            "liquidity_score": "mean",
        })
    )
    collateral_model["collateral_ratio"] = collateral_model["effective_collateral"] / collateral_model["exposure_before_collateral"]
    collateral_model["margin_call_flag"] = (collateral_model["net_exposure"] > 0).astype(int)
    
    output_path = os.path.join(DATA_DIR, 'dashboard', "collateral_daily_model.csv")
    collateral_model.to_csv(output_path, index=False)
    print(f"Model dataset ready: {len(collateral_model):,} rows -> {output_path}")

def merge_collateral_with_macro():
    """Merge collateral data with macro data"""
    collateral = pd.read_csv(os.path.join(DATA_DIR, "collateral_daily_detailed.csv"))
    macro = pd.read_csv(os.path.join(DATA_DIR, "macro_data.csv"))
    
    collateral["date"] = pd.to_datetime(collateral["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    
    collateral["month_year"] = collateral["date"].dt.to_period("M").astype(str)
    macro["month_year"] = macro["date"].dt.to_period("M").astype(str)
    
    merged = collateral.merge(
        macro.drop(columns=["date"]),
        on="month_year",
        how="left"
    ).drop(columns=["month_year"])
    
    out_path = os.path.join(DATA_DIR, "final", "collateral.csv")
    merged.to_csv(out_path, index=False)
    
    print(f"Collateral with macro merged -> {out_path}")
    print(f"Rows: {len(merged):,} | Columns: {len(merged.columns)}")

def main():
    print("Starting collateral processing...")
    
    # Create aggregated model
    create_collateral_model()
    
    # Merge with macro data
    merge_collateral_with_macro()
    
    print("Collateral processing completed")

if __name__ == "__main__":
    main()