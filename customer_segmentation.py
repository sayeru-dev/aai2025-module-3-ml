from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data source: synthetic teaching dataset generated in this project with NumPy seed 42; no real customer records.
ROOT = Path(__file__).resolve().parent
(ROOT / "outputs").mkdir(exist_ok=True)
data = pd.read_csv(ROOT / "data" / "customer_segments.csv")
numeric = ["annual_spending", "purchase_frequency", "age"]
preprocessor = ColumnTransformer([("numeric", StandardScaler(), numeric), ("region", OneHotEncoder(handle_unknown="ignore"), ["region"])])
ready = preprocessor.fit_transform(data)
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(ready).inertia_ for k in range(1, 7)]
plt.figure(figsize=(7, 4))
plt.plot(range(1, 7), inertias, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Customer Segmentation")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "elbow_plot.png", dpi=150)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
data["cluster"] = model.fit_predict(ready)
summary = data.groupby("cluster")[numeric].mean().round(2)
average_spending = data["annual_spending"].mean()
summary["suggested_strategy"] = ["Offer premium rewards" if row["annual_spending"] > average_spending else "Use introductory promotions" for _, row in summary.iterrows()]
summary.to_csv(ROOT / "outputs" / "cluster_summary.csv")
data.to_csv(ROOT / "outputs" / "customer_cluster_assignments.csv", index=False)
text = f"Records: {len(data)}\nClusters used: 3\n\n{summary.to_string()}\n"
(ROOT / "outputs" / "customer_segmentation_results.txt").write_text(text, encoding="utf-8")
print(text)
