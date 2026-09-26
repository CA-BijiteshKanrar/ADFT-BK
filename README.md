# Financial Transaction Anomaly Detection

This project runs in one [Google Colab notebook](https://colab.research.google.com/github/CA-BijiteshKanrar/ADFT-BK/blob/main/Financial_Transaction_Anomaly_Detection.ipynb). It compares IQR, Isolation Forest, Local Outlier Factor, and PCA reconstruction error on the ULB/Worldline credit-card fraud dataset, with a supervised Random Forest benchmark.

## Run in Colab

1. Open the notebook from the link above and select **Runtime > Run all**.
2. The setup cell shallow-clones this public repository's `main` branch into `/content/ADFT-BK` when a new runtime starts. Running the notebook again in the same runtime reuses that clone. All analysis functions are already in the notebook; it does not import code from the clone.
3. Authorize the Google Drive mount when Colab asks. The notebook creates `MyDrive/Financial_Transaction_Anomaly_Detection` by default. Edit `PROJECT_DIR` in the setup cell to use a different Drive folder.
4. For the ULB dataset, place `creditcard.csv` in that Drive folder, or upload it to `/content/creditcard.csv`. If neither file exists, the notebook tries [OpenML dataset 1597](https://www.openml.org/d/1597). The CSV is not stored in this repository.
5. Check `source_info["is_synthetic"]` in the notebook output. It must be `False` for results from the real dataset.

The notebook installs its Python dependencies in the Colab runtime. The default `ALLOW_SYNTHETIC_SMOKE_TEST = False` prevents synthetic data from silently replacing the real dataset. Set it to `True` only for a pipeline smoke test, and set `FAST_MODE = True` to shorten that test.

## Saved results

The notebook writes results to `MyDrive/Financial_Transaction_Anomaly_Detection/real_data` for the real dataset, or to `synthetic_smoke` for a synthetic test. Each run folder contains:

- `figures/`: exploratory plots, validation curves, and the confusion matrix.
- `reports/`: dataset source information, test comparison, and operating summary.
- `artifacts/fraud_detection_bundle.joblib`: the selected detector, scaler, feature order, and locked threshold.

Synthetic results are kept in a separate folder and must not be reported as real-data findings. The notebook also demonstrates scoring one transaction directly. Google Drive preserves saved results after the Colab runtime ends; the clone under `/content` is temporary.

## Dataset and limits

Dataset: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains 284,807 transactions over two days, including 492 labeled frauds. The notebook uses a time-ordered train, validation, and test split, selects alert thresholds on validation data, and evaluates locked thresholds on test data.

This historical, anonymized benchmark demonstrates methodology. Its metrics alone do not establish readiness for autonomous transaction blocking.
