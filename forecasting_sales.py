from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Data source: synthetic monthly housing-demand teaching dataset generated in this project with NumPy seed 42; no real market records.
ROOT = Path(__file__).resolve().parent
(ROOT / "outputs").mkdir(exist_ok=True)
data = pd.read_csv(ROOT / "data" / "housing_demand.csv", parse_dates=["month"])
data["month_index"] = range(len(data))
model = LinearRegression().fit(data[["month_index"]], data["demand"])
data["fitted_demand"] = model.predict(data[["month_index"]])
future = pd.DataFrame({"month_index": range(len(data), len(data) + 6)})
future["month"] = pd.date_range(data["month"].max() + pd.offsets.MonthBegin(1), periods=6, freq="MS")
future["predicted_demand"] = model.predict(future[["month_index"]]).round(1)
future.to_csv(ROOT / "outputs" / "housing_demand_forecast.csv", index=False)
plt.figure(figsize=(8, 4))
plt.plot(data["month"], data["demand"], label="Historical demand")
plt.plot(data["month"], data["fitted_demand"], label="Fitted trend")
plt.plot(future["month"], future["predicted_demand"], "o--", label="Six-month forecast")
plt.xlabel("Month")
plt.ylabel("Demand index")
plt.title("Housing Demand Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "housing_demand_forecast.png", dpi=150)
results = f"Historical records: {len(data)}\nTraining MAE: {mean_absolute_error(data['demand'], data['fitted_demand']):.2f}\nSix-month forecast:\n{future[['month', 'predicted_demand']].to_string(index=False)}\n\nAssumptions: demand follows a simple linear trend and seasonal effects are not modeled. Improvements could include more years of data, seasonality features, and time-series models."
(ROOT / "outputs" / "forecasting_results.txt").write_text(results, encoding="utf-8")
print(results)
