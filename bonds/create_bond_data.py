import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
import os
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(DATA_DIR, "synthetic_bond.csv")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")


# =========================
# BOND GENERATION
# =========================
def create_synthetic_bonds():

    data_path = os.path.join(DATA_DIR, "corporate_fundamentals.csv")
    data = pd.read_csv(data_path)

    data = data.drop_duplicates(subset=["ticker"])

    data["marketCap"] = pd.to_numeric(
        data["marketCap"],
        errors="coerce"
    )

    data["revenue"] = pd.to_numeric(
        data["revenue"],
        errors="coerce"
    )

    data["totalDebt"] = pd.to_numeric(
        data.get("totalDebt"),
        errors="coerce"
    )

    data = data.dropna(
        subset=["marketCap", "revenue"]
    )

    # =========================
    # CREDIT RATING
    # =========================

    def assign_rating(row):

        m = row["marketCap"]

        if m > 500e9:
            return "AA"
        elif m > 100e9:
            return "A+"
        elif m > 20e9:
            return "A"
        elif m > 5e9:
            return "A-"
        elif m > 2e9:
            return "BBB+"
        else:
            return "BBB"

    data["credit_rating"] = data.apply(
        assign_rating,
        axis=1
    )

    # =========================
    # RATES
    # =========================

    fred = Fred(api_key=FRED_API_KEY)

    base_yield = fred.get_series(
        "DGS10",
        observation_start="2022-01-01",
        observation_end="2025-09-30"
    ).mean()

    spread_by_rating = {
        "AA": 0.70,
        "A+": 0.90,
        "A": 1.10,
        "A-": 1.40,
        "BBB+": 1.70,
        "BBB": 2.10,
    }

    np.random.seed(42)

    rows = []

    # =========================
    # GENERATION
    # =========================

    for _, r in data.iterrows():

        ticker = r["ticker"]

        rating = r["credit_rating"]

        spread = spread_by_rating.get(rating, 2.5)

        corp_yield = base_yield + spread

        # =========================
        # CAPITAL STRUCTURE
        # =========================

        market_cap = float(r["marketCap"])

        revenue = float(r["revenue"])

        total_debt = float(
            r.get("totalDebt", 0) or 0
        )

        # fallback if debt unavailable
        if total_debt <= 0:
            total_debt = revenue * 0.35

        # =========================
        # ISSUE SIZE
        # =========================

        issue_size = min(
            total_debt * np.random.uniform(0.20, 0.70),
            market_cap * 0.60
        )

        issue_size = max(issue_size, 100e6)

        # =========================
        # BOND UNITS
        # =========================

        face_value_per_bond = 1000

        units_issued = (
            issue_size / face_value_per_bond
        )

        outstanding_pct = np.random.uniform(
            0.70,
            1.00
        )

        units_outstanding = (
            units_issued * outstanding_pct
        )

        # =========================
        # DATES
        # =========================

        issue_date = (
            datetime(2023, 7, 1)
            + timedelta(days=np.random.randint(0, 365))
        )

        maturity_date = (
            issue_date
            + timedelta(
                days=np.random.randint(
                    5 * 365,
                    15 * 365
                )
            )
        )

        maturity_years = (
            maturity_date - issue_date
        ).days / 365

        # =========================
        # COUPON
        # =========================

        coupon_rate = round(
            np.random.uniform(0.02, 0.06),
            4
        )

        # =========================
        # MARKET PRICE
        # =========================

        # quoted as % of par
        bond_price = 100 + np.random.normal(0, 2)

        # =========================
        # MARKET VALUE
        # =========================

        market_value = (
            (bond_price / 100)
            * face_value_per_bond
            * units_outstanding
        )

        # =========================
        # ROW
        # =========================

        rows.append({

            # IDENTIFIERS
            "bond_id":
                f"{ticker}_{issue_date.strftime('%Y%m%d')}",

            "ticker": ticker,

            "sector": r.get("sector"),

            "industry": r.get("industry"),

            # CREDIT
            "credit_rating": rating,

            # RATES
            "benchmark_yield":
                round(base_yield, 2),

            "corporate_yield":
                round(corp_yield, 2),

            "credit_spread":
                round(spread, 2),

            "yield_to_maturity":
                round(corp_yield, 2),

            # STRUCTURE
            "issue_size":
                round(issue_size, 2),

            "face_value_per_bond":
                face_value_per_bond,

            "units_issued":
                round(units_issued),

            "units_outstanding":
                round(units_outstanding),

            "outstanding_pct":
                round(outstanding_pct, 4),

            # TERMS
            "coupon_rate":
                coupon_rate,

            "bond_price":
                round(bond_price, 2),

            "market_value":
                round(market_value, 2),

            # TIME
            "issue_date":
                issue_date.date(),

            "maturity_date":
                maturity_date.date(),

            "maturity_years":
                round(maturity_years, 1),
        })

    return pd.DataFrame(rows)


# =========================
# CREDIT METRICS
# =========================

def calculate_credit_metrics(df):

    RECOVERY_RATE = 0.40

    rating_pd = {
        "AA": 0.0003,
        "A+": 0.0005,
        "A": 0.0010,
        "A-": 0.0020,
        "BBB+": 0.0050,
        "BBB": 0.0100,
    }

    def spread_to_pd(spread):

        h = (
            (spread / 100)
            / (1 - RECOVERY_RATE)
        )

        pd_val = 1 - np.exp(-h)

        return h, pd_val

    def map_rating(pd_val):

        return min(
            rating_pd,
            key=lambda r: abs(
                pd_val - rating_pd[r]
            )
        )

    hazard_list = []
    pd_list = []
    rating_list = []

    for _, row in df.iterrows():

        h, pd_val = spread_to_pd(
            row["credit_spread"]
        )

        hazard_list.append(h)

        pd_list.append(pd_val)

        rating_list.append(
            map_rating(pd_val)
        )

    df["implied_hazard"] = hazard_list

    df["implied_pd"] = pd_list

    df["implied_rating"] = rating_list

    return df


# =========================
# MAIN
# =========================

def main():

    print("Creating synthetic bonds...")

    df = create_synthetic_bonds()

    print("Calculating credit metrics...")

    df = calculate_credit_metrics(df)

    print("Saving...")

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    df.to_csv(OUTPUT, index=False)

    print(df.head(3))

    print("Total bonds:", len(df))

    print(
        "Total market value:",
        f"${df['market_value'].sum():,.0f}"
    )


if __name__ == "__main__":
    main()