# Module 3 Machine Learning Coding Assignment

This folder contains the three required machine-learning exercises and the
optional housing-demand forecast. Each dataset has at least 100 records.

## Data source

The four CSV files are ChatGPT-generated synthetic teaching datasets created
with a fixed random seed in `generate_datasets.py`. This follows the dataset-
generation option in the instructor's Module 3 rubric. They do not contain real
customer or property information.

## Files

- `house_price_prediction.py`: linear regression using square footage and a
  one-hot encoded location.
- `customer_churn_prediction.py`: logistic regression using scaled numerical
  features and a one-hot encoded region.
- `customer_segmentation.py`: scaled K-Means clustering, elbow plot, cluster
  summary, strategies, and saved assignments.
- `forecasting_sales.py`: optional six-month housing-demand forecast and plot.
- `generate_datasets.py`: reproducibly creates all source CSV files.

## Run

```bash
python -m pip install -r requirements.txt
python generate_datasets.py
python house_price_prediction.py
python customer_churn_prediction.py
python customer_segmentation.py
python forecasting_sales.py
```

The plots and generated result CSV files are saved in `outputs/`.

For submission, upload this entire folder to GitHub or Colab and submit the
working link.
