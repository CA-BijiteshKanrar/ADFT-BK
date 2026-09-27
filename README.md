# Financial Transaction Anomaly Detection

This project runs in one [Google Colab notebook](https://colab.research.google.com/drive/11jos1zldb7tFKxYZ7Pji5wCb0e4CcUCU#scrollTo=4b4bc0ed49fe&uniqifier=1). It compares IQR, Isolation Forest, Local Outlier Factor, and PCA reconstruction error on the ULB/Worldline credit-card fraud dataset, with a supervised Random Forest benchmark.

## Run in Colab

1. Open the notebook from the link above and select **Runtime > Run all**.
2. The setup cell shallow-clones this public repository's `main` branch into `/content/ADFT-BK` when a new runtime starts. A later run reuses the clone; if it predates the bundled dataset, the setup cell updates it with `git pull --ff-only`. All analysis functions are embedded in the notebook.
3. The submitted notebook mounts the evaluator's own Google Drive and writes outputs under `/content/drive/MyDrive/Financial_Transaction_Anomaly_Detection`. It does not require access to the author's Drive.
4. The notebook reads `content/creditcard.csv.zip` from the clone. No dataset upload is needed. If the ZIP is unavailable, it checks `/content/creditcard.csv`, then [OpenML dataset 1597](https://www.openml.org/d/1597).
5. Check `source_info["is_synthetic"]` in the notebook output. It must be `False` for results from the real dataset.

The notebook installs its Python dependencies in the Colab runtime. The default `ALLOW_SYNTHETIC_SMOKE_TEST = False` prevents synthetic data from silently replacing the real dataset. Set it to `True` only for a pipeline smoke test, and set `FAST_MODE = True` to shorten that test.

## Saved results

The notebook writes results to `/content/drive/MyDrive/Financial_Transaction_Anomaly_Detection/real_data` for the real dataset, or to `synthetic_smoke` for a synthetic test. Each run folder contains:

- `figures/`: exploratory plots, validation curves, and the confusion matrix.
- `reports/`: dataset source information, test comparison, and operating summary.
- `artifacts/fraud_detection_bundle.joblib`: the selected detector, scaler, feature order, and locked threshold.

Synthetic results are kept in a separate folder and must not be reported as real-data findings. The notebook also demonstrates scoring one transaction directly. Drive authorization is required to run the notebook's setup cell and save those outputs.

## Dataset and limits

Dataset: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), created by the Machine Learning Group at ULB and Worldline. Kaggle lists the database under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/) and its individual contents under the [Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/). The supplied ZIP was copied unchanged to `content/creditcard.csv.zip` (SHA-256: `d9ade8e5a6c7b39cf3afef2de0fc10ecf9337c2cebdc6cc59a54b1d9dce007cc`). It contains 284,807 transactions over two days, including 492 labeled frauds. The notebook uses a time-ordered train, validation, and test split, selects alert thresholds on validation data, and evaluates locked thresholds on test data.

This historical, anonymized benchmark demonstrates methodology. Its metrics alone do not establish readiness for autonomous transaction blocking.

## Streamlit demonstration

**Live app:** [Financial Transaction Review](https://ccfraudrisk.streamlit.app/)

The app serves two fitted models. Choose between the supervised **SMOTE + class-weighted Random Forest** and the validation-selected **PCA reconstruction anomaly detector** in the sidebar. The Random Forest displays a fraud probability; PCA displays a reconstruction-error anomaly score. Both use a review threshold selected on validation data. Their fitted artifacts are included in `models/`, so opening the app does not retrain them or require Google Drive.

| App tab | What it does |
| --- | --- |
| **CSV Import** | Click **Evaluate bundled test data** to score the 42,559 held-out test rows, view confusion counts and metrics, and download the scored results. Alternatively, download the sample CSV or upload up to 10,000 transactions for scoring. |
| **Single transaction** | Choose an example, edit `Time`, `Amount`, and the anonymized `V1`–`V28` values, then score it. |
| **Model comparison** | View the locked test metrics stored with both packaged model artifacts. |

Uploaded CSVs need numeric `Time`, `Amount`, and `V1`–`V28` columns. Extra columns, including `Class`, are ignored during scoring. The known `Class` labels in the bundled test split are used only to calculate evaluation metrics. A flag is a **manual-review recommendation**, not proof of fraud or an automatic payment block.

To run locally with Python 3.12:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The deployed app runs from `streamlit_app.py` on the repository's `main` branch with Python 3.12 and the root `requirements.txt`. Its bundled model artifacts, example transactions, and dataset are included in the repository. The hosted app needs no Google Drive connection, API key, or dataset upload to evaluate the held-out split.

To rebuild the app's two artifacts from the bundled dataset, run `python train_streamlit_models.py --model both`. This follows the notebook's cleaning, feature order, chronological split, validation threshold rule, and Random Forest settings. The app's anomaly search uses a smaller Isolation Forest grid for repeatable artifact training; PCA was the selected anomaly model in the packaged run. The app artifacts are separate from the notebook's saved Isolation Forest/LOF inference bundle, and retraining can change metrics across library versions.

This public demo uses a historical, anonymized two-day benchmark. Its offline results do not establish live fraud-detection performance. Do not upload real customer data or use its alerts to block payments.
