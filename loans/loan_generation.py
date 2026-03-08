import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    # === LOAD INPUTS ===
    unique_sym = pd.read_csv(os.path.join(DATA_DIR, "unique_sym_sector_industry.csv"))
    bonds = pd.read_csv(os.path.join(DATA_DIR, "final", "bonds.csv"))
    fx = pd.read_csv(os.path.join(DATA_DIR, "final", "fx.csv"))

    start_date = datetime(2024, 7, 1)
    end_date = datetime(2025, 9, 30)

    # Get company metadata from unique_sym data
    company_meta = unique_sym[
        ['ticker', 'sector', 'industry']
    ].drop_duplicates('ticker')

    rows = []

    for _, firm in company_meta.iterrows():
        ticker, sector, industry = firm['ticker'], firm['sector'], firm['industry']

        bond_sub = bonds[bonds['ticker'] == ticker]
        if bond_sub.empty:
            continue

        avg_spread = bond_sub['credit_spread'].mean()
        avg_coupon = bond_sub['coupon_rate'].mean()

        n_loans = np.random.randint(2, 5)

        for i in range(n_loans):
            loan_id = f"{ticker}_L{i+1}"
            issue_date = start_date - timedelta(days=np.random.randint(180, 540))
            maturity_date = issue_date + timedelta(days=np.random.randint(2*365, 7*365))

            fx_currency = "USD"
            fx_row = fx.sample(1).iloc[0]
            fx_rate = fx_row.get("fx_rate", 1.0)

            notional_oc = np.random.uniform(10_000_000, 500_000_000)
            notional_usd = notional_oc * fx_rate
            spread_bps = avg_spread + np.random.uniform(-30, 30)
            coupon_rate = avg_coupon + np.random.uniform(-0.5, 0.5)
            rate_type = np.random.choice(["Fixed", "Floating"], p=[0.3, 0.7])

            for month_offset in range(15):
                snapshot_date = start_date + pd.DateOffset(months=month_offset)
                if snapshot_date > end_date or snapshot_date.date() > maturity_date.date():
                    break

                rows.append({
                    "loan_id": loan_id,
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "currency": fx_currency,
                    "issue_date": issue_date.date(),
                    "maturity_date": maturity_date.date(),
                    "spread_bps": round(spread_bps, 2),
                    "coupon_rate": round(coupon_rate, 3),
                    "rate_type": rate_type,
                    "notional_oc": round(notional_oc, 2),
                    "fx_rate_asof": round(fx_rate, 4),
                    "notional_usd": round(notional_usd, 2),
                    "date": snapshot_date.date(),
                })

    loan_base = pd.DataFrame(rows)
    output_path = os.path.join(DATA_DIR, "loan_synthetic_base.csv")
    loan_base.to_csv(output_path, index=False)
    print(f"Loan base dataset created: {loan_base.shape[0]} rows, {loan_base.shape[1]} cols")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    main()