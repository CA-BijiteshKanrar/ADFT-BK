# Model Card: Financial Transaction Anomaly Detector

## Intended use

Prioritize suspicious credit-card transactions for manual fraud review and
demonstrate advanced anomaly-detection techniques in an academic project.

## Out-of-scope use

- Autonomous denial, account blocking, or adverse customer action.
- Use as proof that a customer committed fraud.
- Direct deployment on a different institution's data without validation.
- Fairness claims based on the absence of demographic attributes.

## Model family

The notebook compares Isolation Forest, Local Outlier Factor, PCA reconstruction
error, an IQR amount baseline, and a supervised Random Forest benchmark. The
stronger service-compatible sklearn detector between Isolation Forest and Local
Outlier Factor is serialized based on validation F2.

## Inputs and output

- Inputs: `Time`, `V1`-`V28`, and `Amount`.
- Output: continuous anomaly score and a binary manual-review recommendation.
- Threshold: selected on validation data by a stated false-negative/false-positive
  cost ratio.

## Evaluation

Use PR-AUC, precision, recall, F2, false-positive rate, confusion counts, and
precision at fixed review capacity. Evaluate on a later time partition and avoid
using accuracy as the principal metric.

## Risks

- Distribution shift and adaptive fraud tactics.
- Delayed or inaccurate fraud labels.
- Customer friction from false positives.
- Under-detection if the threshold or review capacity is poorly calibrated.
- Limited explainability because most benchmark features are anonymized.
- Unknown demographic disparities because protected attributes are unavailable.

## Human oversight

The system recommends review; a trained investigator and approved policy decide
the action. Store score, threshold, model version, and reviewer outcome.

## Retraining triggers

Reassess when data-quality rules fail, score distributions drift, alert volume
exceeds capacity, matured-window recall/precision deteriorates, fraud typologies
change, or the cost assumptions change.
