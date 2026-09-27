from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Data source: synthetic teaching dataset generated in this project with NumPy seed 42; no real property records.
ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "data" / "house_prices.csv")
X = data[["square_footage", "location"]]
y = data["price"]
preprocessor = ColumnTransformer([("location", OneHotEncoder(handle_unknown="ignore"), ["location"])], remainder="passthrough")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(preprocessor.fit_transform(X_train), y_train)
predictions = model.predict(preprocessor.transform(X_test))
new_house = pd.DataFrame({"square_footage": [2000], "location": ["Downtown"]})
new_price = model.predict(preprocessor.transform(new_house))[0]
metrics = f"Records: {len(data)}\nMAE: ${mean_absolute_error(y_test, predictions):,.2f}\nR2: {r2_score(y_test, predictions):.3f}\nPredicted price for a 2,000 sq ft Downtown house: ${new_price:,.2f}\n"
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "house_price_results.txt").write_text(metrics, encoding="utf-8")
pd.DataFrame({"actual_price": y_test.values, "predicted_price": predictions}).to_csv(ROOT / "outputs" / "house_price_predictions.csv", index=False)
print(metrics)
