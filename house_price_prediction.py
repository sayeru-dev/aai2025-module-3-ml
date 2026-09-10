"""Part 1: predict a house price with linear regression."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Data source: ChatGPT-generated synthetic teaching dataset (180 records),
# created with a fixed seed in generate_datasets.py as allowed by the rubric.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "house_prices.csv"


def main() -> None:
    df = pd.read_csv(DATA_FILE).dropna()
    if len(df) < 100:
        raise ValueError("The assignment requires at least 100 house records.")

    X = df[["square_footage", "location"]]
    y = df["price"]

    # Downtown is dropped and becomes the comparison (baseline) location.
    preprocessor = ColumnTransformer(
        [
            (
                "location",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                ["location"],
            ),
            ("size", "passthrough", ["square_footage"]),
        ]
    )
    model = Pipeline(
        [("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)

    new_house = pd.DataFrame(
        {"square_footage": [2000], "location": ["Downtown"]}
    )
    predicted_price = model.predict(new_house)[0]
    test_r2 = r2_score(y_test, model.predict(X_test))

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = dict(
        zip(feature_names, model.named_steps["regressor"].coef_)
    )
    sqft_effect = coefficients["size__square_footage"]

    print(f"Records used: {len(df)}")
    print(f"Test R-squared: {test_r2:.3f}")
    print(
        "Predicted price for a 2000 sq ft house in Downtown: "
        f"${predicted_price:,.2f}"
    )
    print("\nCoefficient explanation:")
    print(
        f"Each additional square foot adds about ${sqft_effect:,.2f} "
        "to price when location stays the same."
    )
    for location in ["Rural", "Suburb"]:
        effect = coefficients[f"location__location_{location}"]
        direction = "higher" if effect >= 0 else "lower"
        print(
            f"A {location} house is about ${abs(effect):,.2f} {direction} than "
            "a Downtown house of the same size."
        )


if __name__ == "__main__":
    main()
