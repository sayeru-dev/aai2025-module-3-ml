"""Create four realistic synthetic datasets with fixed random seeds.

Data source: ChatGPT-generated synthetic teaching data, using the dataset-
generation option described in the instructor's Module 3 rubric. The data are
not real customer or property records.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RNG = np.random.default_rng(42)


def make_house_prices() -> None:
    """Create 180 house records with size, location, and price."""
    count = 180
    square_footage = RNG.integers(650, 4201, count)
    location = RNG.choice(
        ["Downtown", "Suburb", "Rural"], count, p=[0.30, 0.48, 0.22]
    )
    location_effect = pd.Series(location).map(
        {"Downtown": 140_000, "Suburb": 55_000, "Rural": -15_000}
    )
    price = (
        65_000
        + square_footage * 190
        + location_effect.to_numpy()
        + RNG.normal(0, 42_000, count)
    )

    houses = pd.DataFrame(
        {
            "square_footage": square_footage,
            "location": location,
            "price": np.maximum(120_000, np.round(price / 1000) * 1000).astype(int),
        }
    )
    houses.to_csv(DATA_DIR / "house_prices.csv", index=False)


def make_customer_churn() -> None:
    """Create 300 customer records with a realistic churn relationship."""
    count = 300
    age = RNG.integers(18, 76, count)
    usage = np.clip(RNG.normal(34, 15, count), 3, 90).round(1)
    purchase_amount = np.clip(RNG.lognormal(5.0, 0.55, count), 25, 700).round(2)
    service_calls = np.clip(RNG.poisson(2.2, count), 0, 10)
    region = RNG.choice(["North", "South", "East", "West"], count)

    region_effect = pd.Series(region).map(
        {"North": 0.05, "South": -0.10, "East": 0.20, "West": 0.10}
    )
    churn_log_odds = (
        -0.70
        - 0.025 * (usage - 34)
        - 0.003 * (purchase_amount - 150)
        + 0.48 * (service_calls - 2)
        - 0.008 * (age - 40)
        + region_effect.to_numpy()
    )
    churn_probability = 1 / (1 + np.exp(-churn_log_odds))
    churn = RNG.binomial(1, churn_probability)

    customers = pd.DataFrame(
        {
            "age": age,
            "monthly_usage_hours": usage,
            "purchase_amount": purchase_amount,
            "customer_service_calls": service_calls,
            "region": region,
            "churn": churn,
        }
    )
    customers.to_csv(DATA_DIR / "customer_churn.csv", index=False)


def make_customer_segments() -> None:
    """Create 240 customers in three visibly different behavior groups."""
    profiles = [
        # count, spending mean, spending sd, frequency mean, frequency sd, age mean
        (80, 750, 220, 5, 2, 40),
        (80, 3_200, 600, 24, 4, 35),
        (80, 6_200, 850, 10, 3, 53),
    ]
    frames = []
    for count, spend_mean, spend_sd, freq_mean, freq_sd, age_mean in profiles:
        frames.append(
            pd.DataFrame(
                {
                    "annual_spending": np.clip(
                        RNG.normal(spend_mean, spend_sd, count), 100, None
                    ).round(2),
                    "purchase_frequency": np.clip(
                        RNG.normal(freq_mean, freq_sd, count), 1, 40
                    ).round().astype(int),
                    "age": np.clip(RNG.normal(age_mean, 8, count), 18, 75)
                    .round()
                    .astype(int),
                    "region": RNG.choice(
                        ["North", "South", "East", "West"], count
                    ),
                }
            )
        )

    customers = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=42
    )
    customers.to_csv(DATA_DIR / "customer_segmentation.csv", index=False)


def make_housing_demand() -> None:
    """Create 120 months of housing sales demand."""
    month = np.arange(1, 121)
    dates = pd.date_range("2016-01-01", periods=len(month), freq="MS")
    seasonality = 28 * np.sin(2 * np.pi * (month - 3) / 12)
    houses_sold = 210 + 2.1 * month + seasonality + RNG.normal(0, 16, len(month))

    demand = pd.DataFrame(
        {
            "month": month,
            "date": dates.strftime("%Y-%m-%d"),
            "houses_sold": np.maximum(1, np.round(houses_sold)).astype(int),
        }
    )
    demand.to_csv(DATA_DIR / "housing_demand.csv", index=False)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    make_house_prices()
    make_customer_churn()
    make_customer_segments()
    make_housing_demand()
    print("Created four datasets in the data folder.")


if __name__ == "__main__":
    main()
