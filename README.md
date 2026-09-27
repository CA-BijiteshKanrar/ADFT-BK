# Financial Transaction Anomaly Detection

This project runs in one [Google Colab notebook](https://colab.research.google.com/github/CA-BijiteshKanrar/ADFT-BK/blob/main/Financial_Transaction_Anomaly_Detection.ipynb). It compares IQR, Isolation Forest, Local Outlier Factor, and PCA reconstruction error on the ULB/Worldline credit-card fraud dataset, with a supervised Random Forest benchmark.

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

The [Streamlit app](streamlit_app.py) serves two pre-fitted models from this repository: the supervised SMOTE + Random Forest benchmark and a validation-selected anomaly detector (PCA reconstruction in the packaged run). In **CSV Import**, click **Evaluate bundled test data** to run the selected model on the packaged dataset's 42,559 held-out test rows, see metrics against their known labels, and download the scored results. No CSV upload is needed for this evaluation. The app also scores one sample transaction or an imported CSV batch of up to 10,000 rows. Its inputs are the raw `Time`, `Amount`, and `V1`–`V28` fields. `Class` is neither required nor used for inference; it is used only to calculate metrics on the held-out test data. A flag means **manual review**, not confirmed fraud.

To run locally with Python 3.12:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io/), select repository `CA-BijiteshKanrar/ADFT-BK`, branch `main`, and entrypoint `streamlit_app.py`; choose Python 3.12. The root `requirements.txt` declares the app dependencies. The model files under `models/`, examples under `samples/`, and dataset under `content/` are included, so the app does not train at startup or need Google Drive, API keys, or dataset uploads.

To rebuild both artifacts from the bundled dataset, run `python train_streamlit_models.py --model both`. This uses the notebook's cleaning, feature order, time split, validation threshold rule, and Random Forest settings. For practical artifact training, the anomaly search uses a smaller Isolation Forest grid than the notebook; the packaged anomaly model is selected on validation F2 after choosing the lowest-cost setting within each family. Rebuilding can produce slightly different numbers with different library versions. The app's Model comparison tab reports the locked test results stored with the artifacts.

This public demo is for anonymized benchmark transactions. Do not upload real customer data or use its alert decision to block a payment.
