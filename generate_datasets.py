from pathlib import Path
import numpy as np
import pandas as pd

# Data source: synthetic teaching datasets generated for this assignment; no real personal, customer, property, or market records.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

n = 240
locations = rng.choice(["Downtown", "Northside", "Southside", "West End"], n)
sqft = rng.integers(800, 3201, n)
effects = {"Downtown": 90000, "Northside": 45000, "Southside": 10000, "West End": 65000}
price = 155 * sqft + np.array([effects[x] for x in locations]) + rng.normal(0, 30000, n)
pd.DataFrame({"price": price.round(0).astype(int), "square_footage": sqft, "location": locations}).to_csv(DATA / "house_prices.csv", index=False)

n = 300
age = rng.integers(18, 76, n)
usage = rng.normal(22, 10, n).clip(1, 60)
purchase = rng.normal(85, 35, n).clip(5, 250)
service_calls = rng.poisson(2, n).clip(0, 10)
regions = rng.choice(["West", "East", "North", "South"], n)
logit = 1.8 - 0.055 * usage - 0.012 * purchase + 0.34 * service_calls + 0.01 * (age - 40) + (regions == "South") * 0.25
prob = 1 / (1 + np.exp(-logit))
churn = rng.binomial(1, prob)
pd.DataFrame({"age": age, "monthly_usage": usage.round(1), "purchase_amount": purchase.round(2), "service_calls": service_calls, "region": regions, "churn": churn}).to_csv(DATA / "customer_churn.csv", index=False)

n = 240
segment = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
spending = np.where(segment == 0, rng.normal(450, 150, n), np.where(segment == 1, rng.normal(1200, 220, n), rng.normal(2600, 350, n)))
frequency = np.where(segment == 0, rng.normal(3, 1.5, n), np.where(segment == 1, rng.normal(9, 2, n), rng.normal(18, 3, n)))
age = rng.integers(18, 76, n)
regions = rng.choice(["West", "East", "North", "South"], n)
pd.DataFrame({"annual_spending": spending.clip(50).round(2), "purchase_frequency": frequency.clip(1).round(1), "age": age, "region": regions}).to_csv(DATA / "customer_segments.csv", index=False)

months = pd.date_range("2023-01-01", periods=120, freq="MS")
trend = np.arange(120) * 3
season = 12 * np.sin(2 * np.pi * np.arange(120) / 12)
demand = 100 + trend + season + rng.normal(0, 5, 120)
pd.DataFrame({"month": months, "demand": demand.round(1)}).to_csv(DATA / "housing_demand.csv", index=False)
print("Created four datasets with 100+ records each.")
