import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
import os
from dotenv import load_dotenv
from math import exp

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(DATA_DIR, "synthetic_bond.csv")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

def create_synthetic_bonds():
    """Create synthetic bond data with credit ratings and spreads"""
    SYM = os.path.join(DATA_DIR, "unique_sym_sector_industry.csv")
    data = pd.read_csv(SYM)

    # === RATING MAPS ===
    top_tier = {
        "AAA": ["MSFT", "AAPL", "JNJ", "PG", "V", "MA", "COST"],
        "AA+": ["GOOG", "AMZN", "NVDA", "META", "HD", "PEP", "XOM", "JPM"],
        "AA": ["UNH", "LLY", "MRK", "KO", "DIS", "CSCO", "ORCL", "INTC", "TXN",
               "PFE", "MCD", "CVX", "BAC", "GS", "MS", "UPS", "CAT", "NKE"]
    }
    rating_map = {t: r for r, tickers in top_tier.items() for t in tickers}
    sector_rating_weights = {
        "Technology": ["AA+", "AA", "A+"],
        "Healthcare": ["AA", "A+", "A"],
        "Financials": ["A+", "A", "A-"],
        "Energy": ["BBB+", "BBB", "A-"],
        "Consumer Defensive": ["AA", "A+", "A"],
        "Consumer Cyclical": ["A", "A-", "BBB+"],
        "Industrials": ["A", "A-", "BBB+"],
        "Utilities": ["BBB+", "BBB", "BBB-"],
        "Real Estate": ["BBB+", "BBB", "A-"],
        "Communication Services": ["A+", "A", "BBB+"],
    }
    default_rating = "BBB"

    ratings = []
    for _, row in data.iterrows():
        tkr = row["ticker"]
        if tkr in rating_map:
            ratings.append(rating_map[tkr])
        else:
            sec = row["sector"]
            ratings.append(np.random.choice(sector_rating_weights.get(sec, [default_rating])))
    data["credit_rating"] = ratings

    # === FRED Treasury Yields ===
    fred = Fred(api_key=FRED_API_KEY)
    benchmark_series = {"1Y": "DGS1", "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}
    treasury_data = pd.DataFrame()
    for tenor, code in benchmark_series.items():
        treasury_data[tenor] = fred.get_series(code, observation_start="2022-01-01", observation_end="2025-09-30")
    benchmark_yield = treasury_data.mean().to_dict()
    base_yield = benchmark_yield["10Y"]

    spread_by_rating = {
        "AAA": 0.40, "AA+": 0.55, "AA": 0.70, "A+": 0.90, "A": 1.10, "A-": 1.40,
        "BBB+": 1.70, "BBB": 2.10, "BBB-": 2.50, "BB+": 3.00, "BB": 3.50, "B": 4.50, "CCC": 6.00,
    }

    # === SYNTHETIC BONDS ===
    np.random.seed(42)
    rows = []
    for _, r in data.iterrows():
        rating = r["credit_rating"]
        spread = spread_by_rating.get(rating, 2.0)
        corp_yield = base_yield + spread

        issue_date = datetime(2023, 7, 1) + timedelta(days=np.random.randint(0, 365))
        maturity_date = issue_date + timedelta(days=np.random.randint(5 * 365, 15 * 365))
        coupon_rate = round(np.random.uniform(0.02, 0.06), 3)
        bond_price = 100 + np.random.uniform(-3, 3)
        maturity_years = (maturity_date - issue_date).days / 365

        rows.append({
            "bond_id": f"{r['ticker']}_{issue_date.strftime('%Y%m%d')}_{maturity_date.strftime('%Y%m%d')}",
            "ticker": r["ticker"], "sector": r["sector"], "industry": r["industry"], "credit_rating": rating,
            "benchmark_yield": round(base_yield, 2), "corporate_yield": round(corp_yield, 2),
            "credit_spread": round(spread, 2), "coupon_rate": coupon_rate,
            "issue_date": issue_date.date(), "maturity_date": maturity_date.date(),
            "maturity_years": round(maturity_years, 1), "bond_price": round(bond_price, 2),
            "yield_to_maturity": round(corp_yield, 2),
        })

    return pd.DataFrame(rows)

def calculate_credit_metrics(bond_df):
    """Calculate implied credit risk metrics"""
    RECOVERY_RATE = 0.40
    LGD = 1 - RECOVERY_RATE

    rating_pd_table = {
        "AAA": 0.0001, "AA+": 0.0002, "AA": 0.0003, "A+": 0.0005,
        "A": 0.0010, "A-": 0.0020, "BBB+": 0.0050, "BBB": 0.0100,
        "BBB-": 0.0200, "BB+": 0.0400, "BB": 0.0700, "B": 0.1500, "CCC": 0.4000,
    }

    def market_spread_to_implied_pd(spread_pct, recovery_rate=RECOVERY_RATE):
        spread_decimal = spread_pct / 100
        hazard = spread_decimal / (1 - recovery_rate)
        implied_pd = 1 - exp(-hazard)
        return hazard, implied_pd

    def map_pd_to_rating(pd_a):
        diffs = {r: abs(pd_a - v) for r, v in rating_pd_table.items()}
        return min(diffs, key=diffs.get)

    def pd_to_score(pd_a):
        alpha = 100.0
        pd_max = 0.4
        raw = 1 - (np.log10(1 + alpha * max(pd_a, 1e-9)) / np.log10(1 + alpha * pd_max))
        return float(np.clip(100 * raw, 0, 100))

    hazard, pd_annual, pd_multi, implied_rating, score = [], [], [], [], []
    for _, row in bond_df.iterrows():
        h, pd_a = market_spread_to_implied_pd(row["credit_spread"])
        m_years = row["maturity_years"]
        hazard.append(h)
        pd_annual.append(pd_a)
        pd_multi.append(1 - np.exp(-h * m_years))
        implied_rating.append(map_pd_to_rating(pd_a))
        score.append(pd_to_score(pd_a))

    bond_df["implied_hazard"] = hazard
    bond_df["implied_pd_annual"] = pd_annual
    bond_df["implied_pd_multi_year"] = pd_multi
    bond_df["implied_rating"] = implied_rating
    bond_df["market_synthetic_score"] = score

    return bond_df

def main():
    print("Creating synthetic bond data...")
    bond_df = create_synthetic_bonds()
    print(f"Created {len(bond_df)} synthetic bonds")
    
    print("Calculating credit metrics...")
    bond_df = calculate_credit_metrics(bond_df)
    
    print("Saving bond data...")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    bond_df.to_csv(OUTPUT, index=False)
    print(f"Bond data saved -> {OUTPUT}")
    print(f"Rows: {len(bond_df)} | Columns: {len(bond_df.columns)}")
    print(bond_df.head(3))

if __name__ == "__main__":
    main()