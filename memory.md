# Project conversation memory

Last updated: 2026-09-27. Repository: `ADFT-BK`.

## Current project state

- Main artifact: `Financial_Transaction_Anomaly_Detection.ipynb`. The bundled ULB/Worldline credit-card dataset is `content/creditcard.csv.zip` (284,807 raw transactions, 492 fraud labels).
- The notebook cleans exact duplicates, engineers `LogAmount`, `HourSin`, and `HourCos`, and makes a chronological 70/15/15 train/validation/test split when all partitions contain both classes.
- Anomaly detectors train on normal training transactions. The notebook compares Isolation Forest, Local Outlier Factor (LOF), PCA reconstruction error, and an IQR amount baseline. It also trains a supervised Random Forest (RF) benchmark using SMOTE when `imbalanced-learn` is available.
- Validation data selects model hyperparameters and alert thresholds. Test data is evaluated using those locked thresholds. Missed fraud and false alert costs are 25 and 1, respectively.
- The latest code change set SMOTE `k_neighbors=3` for the supervised RF benchmark. It was pushed to `origin/main` in commit `da7747a` (`Set SMOTE neighbors to 3 for supervised RF benchmark`). The repository was clean after the push. No later code change is recorded in this conversation.

## Conversation log and decisions

1. **Dataset and model settings.** The user asked for the dataset hyperparameters. We explained that the dataset itself has none; the notebook sets a 70/15/15 split, seed 42, anomaly model search grids, RF/SMOTE parameters, and threshold costs.
2. **PCA search.** The user asked for every PCA reconstruction setting. The default run searches retained variance `0.90`, `0.95`, and `0.99`; uses PCA `svd_solver="full"`, seed 42, and up to 100,000 normal training rows. The alert threshold is chosen on validation data using the 25:1 cost ratio.
3. **RF settings.** The user asked for the RF search. We clarified that there is no RF hyperparameter search: the default run uses 400 trees, depth 12, minimum leaf size 2, balanced subsample class weights, seed 42, and all cores. SMOTE originally used a 0.05 sampling strategy and five neighbors.
4. **Three-neighbor experiment.** The user asked what happens with SMOTE `k_neighbors=3`. We ran the bundled data locally with the notebook's cleaning, features, chronological split, seed, model parameters, and validation-based threshold selection. The run had 283,726 cleaned rows; train/validation/test sizes were 198,608/42,559/42,559, with 366/55/52 frauds. With `k=3`, test results were TP 40, FN 12, FP 17, TN 42,490, precision 0.70175, recall 0.76923, F2 0.75472, PR AUC 0.77967, ROC AUC 0.96742, and threshold 0.45150. A same-environment `k=5` comparison had TP 40, FN 12, FP 29, TN 42,478, precision 0.57971, recall 0.76923, F2 0.72202, PR AUC 0.77550, ROC AUC 0.97088, and threshold 0.39638. These are results of that local run, not guaranteed results for every Colab environment.
5. **Code finalization.** At the user's request, we fetched and fast-forwarded `main` to `origin/main`, changed the notebook's SMOTE `k_neighbors` from 5 to 3, verified the notebook JSON and one-line diff, committed, and pushed commit `da7747a`. This is the boundary used for `post_finalization_queries.json`.
6. **PCA after reduction.** The user asked which classifier follows PCA. We explained that there is no separate classifier. The detector computes mean squared reconstruction error and compares it with a validation-selected threshold.
7. **Why PCA is reported as best anomaly model.** The user supplied screenshots of the selection cell and validation table. The cell constructs `anomaly_names = [*anomaly_models.keys(), "IQR Amount Baseline"]`, excluding supervised RF. PCA's screenshot validation F2 was 0.80224, the highest among the included anomaly candidates. RF's validation F2 was higher at 0.82721, so RF would win an overall-model comparison. The printed phrase means best *anomaly detector within that restricted list*.
8. **Unlabeled test transactions.** We explained that RF needs labels for training but can predict on unlabeled transactions. Validation evidence favors RF for an overall model choice, but without test labels no model's test performance can be measured. If training labels are unavailable, the notebook's supervised RF cannot be trained as written.
9. **RF deployment.** We explained that RF is technically deployable, but the notebook's current deployment cell considers only Isolation Forest and LOF. RF inference would need to save the fitted SMOTE/RF pipeline, apply the same engineered features in the same order, call `predict_proba()`, and compare the fraud probability with the validation-selected threshold. The RF was trained on unscaled engineered features; its inference path should not apply the anomaly models' `RobustScaler`.

## Notes for future work

- Keep the distinction between **best anomaly detector** and **best model overall** explicit in reports and notebook output.
- The screenshot values in item 7 came from a Colab validation table and should not be presented as test metrics. They differ from the separate local experiment in item 4.
- If RF deployment or an overall-model selection is requested, update the notebook code and validate the resulting inference path before reporting it as implemented.
- `post_finalization_queries.json` holds the user's questions after commit `da7747a` for use in a detailed report. The entries are summaries, not a verbatim transcript.
