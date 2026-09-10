"""Part 2: predict customer churn with logistic regression."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Data source: ChatGPT-generated synthetic teaching dataset (300 records),
# created with a fixed seed in generate_datasets.py as allowed by the rubric.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "customer_churn.csv"


def main() -> None:
    df = pd.read_csv(DATA_FILE).dropna()
    if len(df) < 100:
        raise ValueError("The assignment requires at least 100 customer records.")

    numerical_features = [
        "age",
        "monthly_usage_hours",
        "purchase_amount",
        "customer_service_calls",
    ]
    categorical_features = ["region"]
    X = df[numerical_features + categorical_features]
    y = df["churn"]

    preprocessor = ColumnTransformer(
        [
            ("numbers", StandardScaler(), numerical_features),
            (
                "region",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    new_customer = pd.DataFrame(
        {
            "age": [35],
            "monthly_usage_hours": [20],
            "purchase_amount": [150],
            "customer_service_calls": [5],
            "region": ["West"],
        }
    )
    churn_probability = model.predict_proba(new_customer)[0, 1]
    threshold = 0.50
    churn_prediction = int(churn_probability >= threshold)

    print(f"Records used: {len(df)}")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    print(f"Churn probability: {churn_probability:.1%}")
    print(f"Prediction at the 0.50 threshold: {churn_prediction}")
    print(
        "Interpretation: this is the estimated chance that the customer will "
        "leave. A business can contact customers classified as 1 with support "
        "or retention offers."
    )

    print("\nModel coefficients (positive values increase churn likelihood):")
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    for feature, coefficient in zip(feature_names, coefficients):
        print(f"{feature}: {coefficient:.3f}")


if __name__ == "__main__":
    main()
