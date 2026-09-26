# Financial Transaction Anomaly Detection

End Course Summative Assignment for **Advanced Machine Learning**.

This repository detects suspicious credit-card transactions under severe class
imbalance. It compares statistical, isolation-based, density-based, and
reconstruction-based anomaly detectors, and includes a supervised benchmark,
cost-sensitive threshold selection, reproducible evaluation, and a deployable
Flask API.

## Project choice

- **Category:** Anomaly Detection in Financial Transactions
- **Domain:** Banking and Finance
- **Primary objective:** Retrieve fraudulent transactions while controlling the
  manual-review burden and the cost of missed fraud.

## Why this solution is technically sound

- Uses a time-ordered 70/15/15 train-validation-test design to approximate
  prospective deployment.
- Fits transformations on training data only.
- Fits anomaly detectors on the normal training profile and clearly describes
  the method as novelty detection.
- Compares IQR, Isolation Forest, Local Outlier Factor, and PCA reconstruction
  error instead of relying on one algorithm.
- Tunes a compact grid for tree count/sample fraction, neighbourhood size, and
  retained PCA variance using validation cost, F2, and PR-AUC.
- Includes a SMOTE plus class-weighted Random Forest benchmark when
  `imbalanced-learn` is available. SMOTE is applied only inside training.
- Avoids accuracy as a selection metric. Reports PR-AUC, ROC-AUC, precision,
  recall, F1, F2, false-positive rate, confusion counts, and precision at a fixed
  review capacity.
- Selects thresholds on validation data by minimizing an explicit missed-fraud
  and false-alert cost. The test set is evaluated once with locked thresholds.
- Packages preprocessing, feature order, model, and threshold together for
  consistent scoring.

## Dataset

Use the anonymized **ULB/Worldline Credit Card Fraud Detection** dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

It contains 284,807 transactions recorded over two days, of which 492 are
labelled fraud. Place `creditcard.csv` at `data/creditcard.csv`. The notebook can
also request OpenML dataset `1597`.

The synthetic fallback is only for verifying that the pipeline executes. Do not
submit synthetic metrics as project results.

## Repository structure

```text
.
|-- Financial_Transaction_Anomaly_Detection.ipynb  # Main collaborative report
|-- PROJECT_REPORT.md                              # Written project report
|-- VIDEO_PRESENTATION_SCRIPT.md                   # <=25-minute narration
|-- PRESENTATION_OUTLINE.md                        # Slide-by-slide plan
|-- MODEL_CARD.md                                  # Governance and limitations
|-- README.md
|-- requirements.txt
|-- app.py                                         # Flask scoring API
|-- Dockerfile
|-- data/README.md
|-- src/fraud_detection.py                         # Shared pipeline utilities
|-- scripts/smoke_test.py
|-- tests/test_pipeline.py
|-- artifacts/                                     # Generated model bundle
|-- figures/                                       # Generated plots
`-- reports/                                       # Generated metrics
```

## Reproduce the analysis

### Option A: Google Colab

1. Upload the repository to GitHub.
2. Open `Financial_Transaction_Anomaly_Detection.ipynb` in Colab.
3. Either upload `creditcard.csv` to `/content/creditcard.csv`, or allow the
   notebook to obtain OpenML dataset 1597.
4. Keep `ALLOW_SYNTHETIC_SMOKE_TEST = False` for the final run.
5. Select **Runtime > Run all**.
6. Confirm `source_info["is_synthetic"]` is `False` before using the results.
7. Share the notebook with view access for everyone and edit access for
   `evaluator@almabetter.com`.

### Option B: Local execution

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Financial_Transaction_Anomaly_Detection.ipynb
```

For a fast code-only validation that does not use the real dataset:

```bash
PYTHONPATH=. python scripts/smoke_test.py
pytest -q
```

## Generated outputs

After executing the notebook on the real dataset:

- `reports/model_comparison_test.csv`
- `reports/operating_summary.json`
- `figures/eda_class_and_amount.png`
- `figures/validation_curves.png`
- `figures/best_anomaly_confusion_matrix.png`
- `artifacts/fraud_detection_bundle.joblib`

## Run the API

First run the notebook to produce the model bundle. Then:

```bash
python app.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 10000,
    "V1": 0.1, "V2": -0.2, "V3": 0.3, "V4": 0.0,
    "V5": 0.1, "V6": 0.0, "V7": -0.1, "V8": 0.2,
    "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0,
    "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0,
    "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
    "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0,
    "Amount": 125.50
  }'
```

The response contains `anomaly_score`, the locked `threshold`, and a decision of
`manual_review` or `allow`. In a real bank this should support human review; it
should not automatically decline customers without policy, testing, and appeal
controls.

## Docker deployment

```bash
docker build -t fraud-anomaly-api .
docker run --rm -p 5000:5000 fraud-anomaly-api
```

## Monitoring and retraining

Monitor schema failures, null rates, score distribution, alert rate, fraud
capture, false-positive customer friction, PR-AUC, recall, investigator capacity,
and label delay. Revisit the operating threshold when fraud loss, investigation
cost, or review capacity changes. Use champion-challenger rollout and retain the
model version, score, threshold, and reviewer outcome for auditability.

## Important limitations

- The benchmark covers only two historical days.
- `V1`-`V28` are anonymized, reducing business explainability.
- Merchant, device, geography, and network relationships are unavailable.
- Protected attributes are absent, so demographic fairness cannot be tested.
- Good benchmark performance is not evidence of readiness for autonomous
  blocking in a contemporary banking environment.

## Attribution

- Dataset: Machine Learning Group, Universite Libre de Bruxelles, *Credit Card
  Fraud Detection*, distributed through Kaggle.
- Course grounding: *Mastering Advanced Machine Learning for Industry
  Applications* and *Mastering ML Infrastructure from Model Training to Scalable
  Deployment*.
- Libraries: scikit-learn, imbalanced-learn, pandas, NumPy, Matplotlib, Seaborn,
  Flask, and joblib.
