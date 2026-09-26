# Project Report: Anomaly Detection in Financial Transactions

## 1. Project identification

- **Course:** Advanced Machine Learning
- **Project category:** Anomaly Detection in Financial Transactions
- **Domain:** Banking and Finance
- **Author:** Bijitesh Kanrar
- **Primary deliverable:** `Financial_Transaction_Anomaly_Detection.ipynb`

## 2. Abstract

Financial institutions process a large volume of legitimate transactions while
fraud represents a rare but costly class. This project develops and compares
multiple anomaly-detection approaches for identifying suspicious credit-card
transactions under extreme class imbalance. The analysis uses the anonymized
ULB/Worldline benchmark and compares an IQR baseline, Isolation Forest, Local
Outlier Factor, and PCA reconstruction error. A supervised Random Forest,
supported by SMOTE within the training partition and class weighting, provides a
label-informed reference.

The methodology uses a time-ordered train-validation-test split, training-only
preprocessing, validation-only threshold selection, and one-time locked test
evaluation. Performance is assessed through PR-AUC, ROC-AUC, precision, recall,
F1, F2, false-positive rate, confusion-matrix counts, and precision at a fixed
review capacity. An explicit cost function distinguishes the cost of missed
fraud from the cost of a false alert. The selected anomaly model is packaged
with its scaler, feature order, and threshold and exposed through a Flask API.

## 3. Business problem

A fraud-control system must detect rare harmful activity without flooding
investigators or inconveniencing legitimate customers. The problem is therefore
not simply to maximize classification accuracy. The operational goal is to rank
transactions by suspicion and select an alert threshold that balances:

- fraud loss prevented;
- manual investigation cost;
- false-positive customer friction;
- available review capacity;
- model latency and maintainability.

This project treats the model as a decision-support control. A suspicious flag
routes the transaction for review rather than automatically proving fraud.

## 4. Dataset

The ULB/Worldline dataset contains 284,807 European credit-card transactions
recorded over two days, including 492 labelled frauds. `V1` through `V28` are
PCA-anonymized variables. `Time` records elapsed seconds, `Amount` records the
transaction value, and `Class` is the binary target.

The fraud rate is approximately 0.17%, making conventional accuracy unsuitable:
a model that predicts every transaction as normal would appear highly accurate
while providing no fraud protection.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 5. Data preprocessing

The pipeline performs the following controlled steps:

1. Validate the required columns and binary target.
2. Convert all required values to numeric data types.
3. Record missing values and exact duplicates before removing unusable rows.
4. Retain legitimate extreme observations instead of deleting them merely for
   being anomalous.
5. Apply `log1p` to `Amount` to reduce right skew.
6. Convert time-of-day into sine and cosine variables so midnight is represented
   as adjacent to the end of the previous day.
7. Fit `RobustScaler` on the training partition only.

Median imputation could be added for a production feed after confirming the
missingness mechanism. The benchmark is expected to be complete; silently
imputing the target is prohibited.

## 6. Experimental design and leakage control

Data is sorted by `Time` and divided into 70% training, 15% validation, and 15%
test partitions. This approximates a real deployment where past transactions are
used to score future ones. A stratified fallback is used only for small
development samples that place one class entirely outside a partition.

The train partition fits preprocessing and estimators. The validation partition
selects thresholds and compares candidates. The test partition is used once
after model and threshold choices are locked. Resampling never occurs before
splitting.

Anomaly detectors are fitted to the training observations labelled normal. This
is **novelty detection**, not completely label-free unsupervised learning. The
distinction is documented because labels are used to construct the clean
reference population, although the algorithms do not optimize against fraud
labels.

## 7. Models

### 7.1 IQR amount baseline

The absolute deviation of log-amount from the training median is divided by the
training interquartile range. This model is transparent and useful as a control,
but it cannot detect multivariate fraud patterns with ordinary amounts.

### 7.2 Isolation Forest

Isolation Forest recursively partitions feature space. Rare observations tend to
be isolated in fewer splits and therefore receive higher anomaly scores. It is
computationally practical for high-dimensional tabular data and requires no fraud
labels in its objective.

### 7.3 Local Outlier Factor

LOF compares each observation's local density with its neighbours. Novelty mode
permits scoring unseen validation and test transactions. It can capture local
abnormality that a global detector misses, although scoring is more expensive.

### 7.4 PCA reconstruction error

PCA learns a lower-dimensional subspace from normal transactions. A transaction
that cannot be reconstructed well from that subspace receives a high mean
squared reconstruction error.

### 7.5 Supervised reference

A Random Forest is trained using class weights. When `imbalanced-learn` is
installed, SMOTE raises the minority share to 5% within the training pipeline
only. The benchmark quantifies the value of labels; it does not replace the
requirement to compare anomaly detectors.

### 7.6 Hyperparameter tuning

A compact grid is evaluated within each anomaly family. Isolation Forest varies
the number of trees and training sample fraction, Local Outlier Factor varies the
number of neighbours, and PCA varies the proportion of variance retained. The
selected configuration in each family minimizes validation business cost, with
F2 and PR-AUC used as tie-breakers. This meets the optimization requirement while
keeping the search auditable and computationally proportionate.

## 8. Threshold selection and evaluation

For each model, higher scores represent greater suspicion. The validation
threshold minimizes:

`Relative cost = 25 x false negatives + 1 x false positives`

The 25:1 ratio is an illustrative assumption. A production bank should replace
it with observed fraud loss, recovery rate, investigation expense, customer
friction, and control appetite.

Reported metrics are:

- **PR-AUC:** emphasizes minority-class ranking quality.
- **Recall:** percentage of frauds captured.
- **Precision:** percentage of alerts that are truly fraudulent.
- **F2:** weights recall more than precision.
- **False-positive rate:** legitimate transactions incorrectly escalated.
- **Precision at 100:** yield if only the top 100 alerts can be reviewed.
- **ROC-AUC:** retained as a secondary ranking measure.

## 9. Results

Execute the notebook on the real ULB dataset. The final results are written to:

- `reports/model_comparison_test.csv`
- `reports/operating_summary.json`

Before submission, paste the executed comparison table and confusion matrix
below or export the notebook with outputs.

> **Submission control:** Results are valid only when
> `source_info["is_synthetic"] == False`. Synthetic smoke-test metrics are not
> evidence about real fraud detection.

Recommended reporting format:

| Model | PR-AUC | Precision | Recall | F2 | FPR | FP | FN | Relative cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQR amount baseline | From notebook | | | | | | | |
| Isolation Forest | From notebook | | | | | | | |
| Local Outlier Factor | From notebook | | | | | | | |
| PCA reconstruction | From notebook | | | | | | | |
| Supervised RF benchmark | From notebook | | | | | | | |

## 10. Deployment design

The notebook serializes a bundle containing:

- selected anomaly detector;
- fitted `RobustScaler`;
- raw and engineered feature order;
- locked validation threshold;
- dataset, split, cost, and performance metadata.

`app.py` exposes:

- `GET /health` for readiness status;
- `POST /predict` for a transaction score and `manual_review`/`allow` decision.

The Dockerfile runs the service through Gunicorn. Production deployment would
add authentication, encryption, rate limiting, schema contracts, centralized
logging, secret management, and a feature service.

## 11. Monitoring and governance

Monitoring should cover data quality, score and feature drift, alert rate,
precision, recall, fraud value captured, false-positive customer friction,
review turnaround, API latency, and error rates. Delayed fraud labels require
performance reporting by matured outcome windows.

Every alert should retain model version, score, threshold, timestamp, and human
review outcome. Model changes should use champion-challenger testing, approval,
rollback, and documented threshold rationale.

Because protected demographic features are absent, this dataset cannot support
a demographic fairness assessment. This is a limitation rather than evidence of
fairness.

## 12. Limitations and future work

- The dataset is historical and covers only two days.
- PCA anonymization reduces business explanation.
- Merchant, device, geography, customer history, and transaction graphs are
  absent.
- A static offline benchmark does not reproduce adaptive adversarial fraud.
- Next steps include rolling temporal validation, graph features, delayed-label
  learning, autoencoders, calibrated probabilities, explanation workflows, and
  live drift monitoring.

## 13. Conclusion

The project provides an end-to-end, reproducible anomaly-detection workflow
grounded in both machine-learning quality and operational banking controls. It
compares multiple outlier approaches, handles imbalance without leaking future
information, selects a business-aware threshold, and demonstrates deployment.
The final model recommendation must be taken from the real-data notebook run,
not chosen in advance.
