"""Extra credit: forecast housing demand for the next six months."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Data source: ChatGPT-generated synthetic monthly housing-demand dataset
# (120 records), created with a fixed seed in generate_datasets.py.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "housing_demand.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["month", "date", "houses_sold"]).sort_values("month")
    if len(df) < 100:
        raise ValueError("The assignment requires at least 100 demand records.")

    X = df[["month"]]
    y = df["houses_sold"]
    model = LinearRegression()
    model.fit(X, y)

    first_future_month = int(df["month"].max()) + 1
    future_months = np.arange(first_future_month, first_future_month + 6)
    future_dates = pd.date_range(
        df["date"].max() + pd.offsets.MonthBegin(1), periods=6, freq="MS"
    )
    predictions = np.maximum(
        0, np.rint(model.predict(pd.DataFrame({"month": future_months})))
    ).astype(int)

    forecast = pd.DataFrame(
        {
            "month": future_months,
            "date": future_dates.strftime("%Y-%m-%d"),
            "predicted_houses_sold": predictions,
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    forecast.to_csv(OUTPUT_DIR / "housing_demand_forecast.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], y, label="Historical demand")
    plt.plot(future_dates, predictions, "--o", label="Six-month forecast")
    plt.xlabel("Date")
    plt.ylabel("Houses sold")
    plt.title("Housing Demand Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "housing_demand_forecast.png", dpi=150)
    plt.close()

    print(f"Records used: {len(df)}")
    print(f"Estimated monthly trend: {model.coef_[0]:.2f} additional sales")
    print("\nSix-month forecast:")
    print(forecast.to_string(index=False))
    print("\nAssumptions: the historical linear trend continues without a major shock.")
    print("Challenge: simple linear regression does not model seasonal changes well.")
    print(
        "Improvement: add interest rates, prices, inventory, and a time-series "
        "model to capture seasonality."
    )


if __name__ == "__main__":
    main()
