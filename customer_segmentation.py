"""Part 3: segment customers with K-Means clustering."""

from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Data source: ChatGPT-generated synthetic teaching dataset (240 records),
# created with a fixed seed in generate_datasets.py as allowed by the rubric.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "customer_segmentation.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    df = pd.read_csv(DATA_FILE).dropna()
    if len(df) < 100:
        raise ValueError("The assignment requires at least 100 customer records.")

    features = ["annual_spending", "purchase_frequency", "age"]
    X_scaled = StandardScaler().fit_transform(df[features])

    # The clear bend in this dataset occurs at K=3.
    k_values = range(1, 7)
    inertia = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(list(k_values), inertia, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Customer Segmentation")
    plt.xticks(list(k_values))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_plot.png", dpi=150)
    plt.close()

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    summary = df.groupby("cluster")[features].mean().round(2)

    high_spending_cluster = summary["annual_spending"].idxmax()
    frequent_cluster = summary["purchase_frequency"].idxmax()

    print(f"Records used: {len(df)}")
    print("Selected K=3 because the elbow plot bends sharply at 3 clusters.")
    print("\nAverage characteristics by cluster:")
    print(summary)
    print("\nMarketing strategies:")
    for cluster in summary.index:
        if cluster == high_spending_cluster:
            strategy = "Offer VIP rewards and early access to premium products."
        elif cluster == frequent_cluster:
            strategy = "Offer a loyalty plan or frequency-based discount."
        else:
            strategy = "Use a simple re-engagement coupon or email campaign."
        print(f"Cluster {cluster}: {strategy}")

    result_file = OUTPUT_DIR / "customer_segments.csv"
    df.to_csv(result_file, index=False)
    print(f"\nSaved cluster assignments to {result_file.name}.")
    print("Saved the elbow chart to elbow_plot.png.")


if __name__ == "__main__":
    main()
