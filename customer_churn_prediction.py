from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data source: synthetic teaching dataset generated in this project with NumPy seed 42; no real customer records.
ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "data" / "customer_churn.csv")
numeric = ["age", "monthly_usage", "purchase_amount", "service_calls"]
features = numeric + ["region"]
X = data[features]
y = data["churn"]
preprocessor = ColumnTransformer([("numeric", StandardScaler(), numeric), ("region", OneHotEncoder(handle_unknown="ignore"), ["region"])])
model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
new_customer = pd.DataFrame({"age": [35], "monthly_usage": [8], "purchase_amount": [25], "service_calls": [5], "region": ["West"]})
new_probability = model.predict_proba(new_customer)[0, 1]
new_class = int(new_probability >= 0.5)
report = classification_report(y_test, predicted, zero_division=0)
results = f"Records: {len(data)}\nAccuracy: {accuracy_score(y_test, predicted):.3f}\nROC AUC: {roc_auc_score(y_test, probabilities):.3f}\nNew customer churn probability: {new_probability:.1%}\nAt-risk classification: {new_class}\n\n{report}"
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "customer_churn_results.txt").write_text(results, encoding="utf-8")
pd.DataFrame({"actual_churn": y_test.values, "predicted_churn": predicted, "churn_probability": probabilities}).to_csv(ROOT / "outputs" / "customer_churn_predictions.csv", index=False)
print(results)
