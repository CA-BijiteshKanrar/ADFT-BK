"""Build the submission notebook without requiring nbformat."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "fraud_detection.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


cells = [
    markdown(
        """
        # Anomaly Detection in Financial Transactions

        **End Course Summative Assignment - Advanced Machine Learning**  
        **Domain:** Banking and Finance  
        **Author:** Bijitesh Kanrar  

        ## Executive summary

        This project develops an operationally realistic fraud-screening pipeline for highly
        imbalanced credit-card transactions. It compares a transparent statistical baseline,
        Isolation Forest, Local Outlier Factor, and PCA reconstruction error. A
        class-imbalance-aware Random Forest is included as a supervised reference where labels
        are available.

        Model selection is based on validation data only. The decision threshold explicitly
        trades the cost of missed fraud against the cost of manual investigation. The locked
        threshold is then evaluated once on a time-ordered test set using PR-AUC, ROC-AUC,
        precision, recall, F1, F2, false-positive rate, and confusion-matrix counts.
        """
    ),
    markdown(
        """
        ## Assessment-criteria mapping

        | Requirement | Evidence in this notebook |
        |---|---|
        | Preprocessing and cleaning | Schema validation, missing-value audit, duplicate removal, robust transformation |
        | Handling imbalance | PR-AUC/F2 metrics, cost-sensitive threshold, normal-only detector training, SMOTE or class-weighted benchmark |
        | Multiple outlier techniques | IQR, Isolation Forest, Local Outlier Factor, PCA reconstruction error |
        | Model comparison | Validation-only tuning and one-time locked test evaluation |
        | Advanced analysis | Time-ordered split, leakage control, top-k review capacity, threshold economics |
        | Practical deployment | Serialized model bundle, Flask API, Dockerfile, monitoring and governance plan |

        ### Research questions

        1. Which anomaly detector retrieves the most fraud while keeping investigation volume manageable?
        2. How much value does labelled supervised learning add over novelty detection?
        3. How should the alert threshold change when missed fraud costs more than a false alert?
        """
    ),
    markdown(
        """
        ## Dataset and attribution

        The final analysis uses the anonymized **ULB/Worldline Credit Card Fraud Detection**
        dataset. It contains European card transactions recorded over two days, including 492
        labelled frauds among 284,807 transactions. `V1`-`V28` are PCA-anonymized attributes;
        `Time`, `Amount`, and `Class` are also provided.

        Source: [Kaggle - Machine Learning Group, ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

        The project does not claim that this historical benchmark represents current banking
        traffic. It is used to demonstrate sound anomaly-detection methodology. The notebook
        may generate synthetic data only when explicitly enabled for a smoke test; synthetic
        metrics must never be submitted as real results.
        """
    ),
    code(
        """
        # Run once in a fresh Google Colab runtime.
        import sys, subprocess

        if "google.colab" in sys.modules:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "imbalanced-learn>=0.12,<1", "seaborn>=0.13,<1"
            ])
        """
    ),
    markdown(
        """
        ## Embedded project functions

        The following cell embeds the reusable project module so this notebook can run as a
        standalone Colab file. The same functions are also retained in `src/fraud_detection.py`
        for the API, tests, and repository workflow.
        """
    ),
    code(MODULE_SOURCE),
    code(
        """
        from pathlib import Path
        import json
        import warnings

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display
        from sklearn.metrics import (
            PrecisionRecallDisplay,
            RocCurveDisplay,
            ConfusionMatrixDisplay,
            average_precision_score,
        )
        from sklearn.preprocessing import RobustScaler

        SEED = 42
        DATA_PATH = Path("data/creditcard.csv")
        ALLOW_SYNTHETIC_SMOKE_TEST = False  # Keep False for the submitted run.
        FAST_MODE = False                   # Set True only for pipeline checks.
        FN_COST = 25.0                      # Relative cost of missed fraud.
        FP_COST = 1.0                       # Relative cost of a manual review.

        for directory in ["artifacts", "reports", "figures"]:
            Path(directory).mkdir(exist_ok=True)

        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_columns", 40)
        """
    ),
    markdown(
        """
        ## 1. Load and verify the data

        The loader first checks `data/creditcard.csv`, then `/content/creditcard.csv`, and then
        OpenML dataset 1597. A synthetic fallback requires explicit permission. This prevents
        accidental submission of fabricated metrics.
        """
    ),
    code(
        """
        raw_df, source_info = load_transactions(
            DATA_PATH,
            allow_openml=True,
            allow_synthetic=ALLOW_SYNTHETIC_SMOKE_TEST,
            synthetic_rows=30_000,
            random_state=SEED,
        )
        display(source_info)
        print(f"Raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

        if source_info["is_synthetic"]:
            warnings.warn(
                "SYNTHETIC SMOKE-TEST DATA IS ACTIVE. Do not submit these outputs.",
                stacklevel=1,
            )
        """
    ),
    code(
        """
        # Data-quality profile before transformation.
        profile = pd.DataFrame({
            "dtype": raw_df.dtypes.astype(str),
            "missing": raw_df.isna().sum(),
            "unique": raw_df.nunique(dropna=False),
        })
        display(profile)
        print("Exact duplicate rows:", int(raw_df.duplicated().sum()))
        display(raw_df.head())
        """
    ),
    markdown(
        """
        ## 2. Clean, explore, and engineer features

        Exact duplicate records and rows with unusable numeric values are removed and reported.
        The target is never imputed. Amount is log-transformed because transaction values are
        strongly right-skewed. Time-of-day is encoded cyclically, avoiding the false assumption
        that hour 23 is far from hour 0.

        No fraud rows are deleted merely because they are anomalous: unusual observations are
        the phenomenon being modelled.
        """
    ),
    code(
        """
        clean_df, cleaning_audit = clean_transactions(raw_df)
        df = engineer_features(clean_df)
        display(cleaning_audit)

        class_summary = (
            df["Class"].value_counts().rename_axis("Class").to_frame("count")
        )
        class_summary["percentage"] = 100 * class_summary["count"] / len(df)
        display(class_summary)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        sns.countplot(data=df, x="Class", ax=axes[0], color="#3465a4")
        axes[0].set_title("Severe class imbalance")
        axes[0].set_yscale("log")

        sns.histplot(data=df, x="Amount", hue="Class", bins=80, element="step",
                     stat="density", common_norm=False, ax=axes[1])
        axes[1].set_xlim(0, df["Amount"].quantile(0.99))
        axes[1].set_title("Amount distribution (to 99th percentile)")

        sns.histplot(data=df, x="LogAmount", hue="Class", bins=60, element="step",
                     stat="density", common_norm=False, ax=axes[2])
        axes[2].set_title("Log-transformed amount")
        plt.tight_layout()
        plt.savefig("figures/eda_class_and_amount.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        """
        ## 3. Leakage-resistant train, validation, and test design

        Transactions are sorted by `Time` and split 70%/15%/15%. This approximates production:
        the model learns from earlier activity and scores later activity. If a small development
        dataset places only one class in a partition, the helper uses a documented stratified
        fallback.

        - **Train:** fit scaler and models.
        - **Validation:** select thresholds and compare models.
        - **Test:** estimate final performance once after all choices are locked.

        RobustScaler is fitted on the training partition only. Unsupervised detectors learn the
        profile of training observations labelled normal; labels are not used inside their loss
        functions. This is novelty detection and should not be mislabeled as fully label-free
        unsupervised training.
        """
    ),
    code(
        """
        split = chronological_split(df, random_state=SEED)
        print("Split strategy:", split.strategy)

        split_profile = pd.DataFrame({
            name: {
                "rows": len(part),
                "frauds": int(part["Class"].sum()),
                "fraud_rate_pct": 100 * part["Class"].mean(),
                "start_time": float(part["Time"].min()),
                "end_time": float(part["Time"].max()),
            }
            for name, part in {
                "train": split.train,
                "validation": split.validation,
                "test": split.test,
            }.items()
        }).T
        display(split_profile)

        scaler = RobustScaler()
        X_train = scaler.fit_transform(split.train[MODEL_FEATURES])
        X_validation = scaler.transform(split.validation[MODEL_FEATURES])
        X_test = scaler.transform(split.test[MODEL_FEATURES])
        y_train = split.train["Class"].to_numpy(dtype=int)
        y_validation = split.validation["Class"].to_numpy(dtype=int)
        y_test = split.test["Class"].to_numpy(dtype=int)
        X_train_normal = X_train[y_train == 0]
        """
    ),
    markdown(
        """
        ## 4. Candidate methods

        1. **IQR amount baseline:** interpretable but univariate and unable to identify fraud with
           an ordinary transaction value.
        2. **Isolation Forest:** isolates rare observations through random feature splits; fast
           and suitable for high-dimensional tabular data.
        3. **Local Outlier Factor (novelty mode):** identifies points with lower local density
           than their neighbours; it can capture local fraud patterns but is heavier at scale.
        4. **PCA reconstruction error:** observations poorly represented by the normal-data
           subspace receive higher error scores.
        5. **Supervised benchmark:** SMOTE is applied to the training partition only, followed by
           a class-weighted Random Forest. If `imbalanced-learn` is unavailable, class weights
           are used without SMOTE.

        A compact hyperparameter grid is evaluated within each anomaly-model family. Isolation
        Forest varies tree count and sample fraction, LOF varies neighbourhood size, and PCA
        varies retained variance. The best configuration in each family minimizes validation
        business cost, with F2 and PR-AUC as tie-breakers. This keeps tuning transparent and
        computationally practical.

        Accuracy is deliberately excluded from model selection because predicting every record
        as normal would exceed 99% accuracy on this dataset while detecting no fraud.
        """
    ),
    code(
        """
        anomaly_models, tuning_table = tune_anomaly_models(
            X_train_normal,
            X_validation,
            y_validation,
            false_negative_cost=FN_COST,
            false_positive_cost=FP_COST,
            fast_mode=FAST_MODE,
            random_state=SEED,
        )
        display(tuning_table.round(5))

        validation_scores = {
            name: anomaly_score(name, model, X_validation)
            for name, model in anomaly_models.items()
        }
        test_scores = {
            name: anomaly_score(name, model, X_test)
            for name, model in anomaly_models.items()
        }

        validation_scores["IQR Amount Baseline"] = iqr_amount_score(
            split.validation, split.train
        )
        test_scores["IQR Amount Baseline"] = iqr_amount_score(split.test, split.train)
        """
    ),
    code(
        """
        supervised_model, imbalance_strategy = fit_supervised_benchmark(
            split.train[MODEL_FEATURES].to_numpy(dtype=float),
            y_train,
            random_state=SEED,
            fast_mode=FAST_MODE,
        )
        print("Supervised imbalance strategy:", imbalance_strategy)
        validation_scores["Supervised RF Benchmark"] = supervised_model.predict_proba(
            split.validation[MODEL_FEATURES].to_numpy(dtype=float)
        )[:, 1]
        test_scores["Supervised RF Benchmark"] = supervised_model.predict_proba(
            split.test[MODEL_FEATURES].to_numpy(dtype=float)
        )[:, 1]
        """
    ),
    markdown(
        """
        ## 5. Validation-only threshold optimization

        A prediction becomes an alert when its score is at or above the threshold. Thresholds
        are selected by minimizing:

        $$\\text{Expected cost}=C_{FN}\\times FN+C_{FP}\\times FP$$

        The illustrative cost ratio is 25:1 because a missed fraudulent transaction is assumed
        to be materially more costly than reviewing a legitimate alert. A bank should replace
        these values with observed fraud loss, recovery, investigation, and customer-friction
        costs.
        """
    ),
    code(
        """
        threshold_choices = {}
        validation_rows = []

        for name, scores in validation_scores.items():
            choice = select_cost_threshold(
                y_validation,
                scores,
                false_negative_cost=FN_COST,
                false_positive_cost=FP_COST,
            )
            threshold_choices[name] = choice
            metrics = classification_metrics(y_validation, scores, choice["threshold"])
            metrics.update({
                "Model": name,
                "Validation_Cost": choice["validation_cost"],
                "Precision_at_100": precision_at_k(y_validation, scores, 100),
            })
            validation_rows.append(metrics)

        validation_table = (
            pd.DataFrame(validation_rows)
            .set_index("Model")
            .sort_values(["F2", "PR_AUC"], ascending=False)
        )
        display(validation_table.round(5))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for name, scores in validation_scores.items():
            PrecisionRecallDisplay.from_predictions(
                y_validation, scores, name=name, ax=axes[0]
            )
            RocCurveDisplay.from_predictions(
                y_validation, scores, name=name, ax=axes[1]
            )
        axes[0].set_title("Validation precision-recall curves")
        axes[1].set_title("Validation ROC curves")
        plt.tight_layout()
        plt.savefig("figures/validation_curves.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        """
        ## 6. Locked test evaluation

        The following cell applies each validation-selected threshold to the untouched test
        partition. This avoids optimistic test-set threshold tuning. For operational selection,
        PR-AUC and F2 are more informative than ROC-AUC under severe class imbalance; false
        positives and review capacity remain essential controls.
        """
    ),
    code(
        """
        test_rows = []
        for name, scores in test_scores.items():
            threshold = threshold_choices[name]["threshold"]
            metrics = classification_metrics(y_test, scores, threshold)
            metrics.update({
                "Model": name,
                "Precision_at_100": precision_at_k(y_test, scores, 100),
                "Estimated_Business_Cost": FN_COST * metrics["FN"] + FP_COST * metrics["FP"],
            })
            test_rows.append(metrics)

        test_table = (
            pd.DataFrame(test_rows)
            .set_index("Model")
            .sort_values(["F2", "PR_AUC"], ascending=False)
        )
        display(test_table.round(5))
        test_table.to_csv("reports/model_comparison_test.csv")
        """
    ),
    code(
        """
        anomaly_names = [*anomaly_models.keys(), "IQR Amount Baseline"]
        best_anomaly_name = max(
            anomaly_names,
            key=lambda name: validation_table.loc[name, "F2"],
        )
        best_threshold = threshold_choices[best_anomaly_name]["threshold"]
        best_predictions = (test_scores[best_anomaly_name] >= best_threshold).astype(int)

        print("Best anomaly detector selected on validation F2:", best_anomaly_name)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            best_predictions,
            display_labels=["Normal", "Fraud"],
            cmap="Blues",
            values_format=",d",
            ax=ax,
        )
        ax.set_title(f"Locked test confusion matrix - {best_anomaly_name}")
        plt.tight_layout()
        plt.savefig("figures/best_anomaly_confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        """
        ## 7. Operational interpretation

        The best statistical model is not automatically the best control. A fraud operations
        team must also consider alert volume, case-handling capacity, latency, stability, and
        explainability. The next cell reports alert workload and fraud capture at the locked
        threshold. Review capacity can also be expressed as a top-$k$ queue.
        """
    ),
    code(
        """
        best_metrics = test_table.loc[best_anomaly_name]
        operating_summary = {
            "selected_anomaly_model": best_anomaly_name,
            "test_transactions": int(len(y_test)),
            "alerts": int(best_metrics["TP"] + best_metrics["FP"]),
            "frauds_captured": int(best_metrics["TP"]),
            "frauds_missed": int(best_metrics["FN"]),
            "legitimate_reviews": int(best_metrics["FP"]),
            "recall": float(best_metrics["Recall"]),
            "precision": float(best_metrics["Precision"]),
            "f2": float(best_metrics["F2"]),
            "pr_auc": float(best_metrics["PR_AUC"]),
            "relative_cost": float(best_metrics["Estimated_Business_Cost"]),
        }
        display(operating_summary)
        save_json(operating_summary, "reports/operating_summary.json")
        """
    ),
    markdown(
        """
        ## 8. Deployment artifact and API

        The API deploys the stronger service-compatible sklearn detector between Isolation
        Forest and Local Outlier Factor, selected on validation F2. PCA remains in the model
        comparison, while the transparent IQR baseline remains a control. The bundle contains
        preprocessing, model, feature order, and locked threshold so training and serving use
        the same logic.
        """
    ),
    code(
        """
        deployment_candidates = ["Isolation Forest", "Local Outlier Factor"]
        deployable_name = max(
            deployment_candidates,
            key=lambda name: validation_table.loc[name, "F2"],
        )
        deployable_threshold = threshold_choices[deployable_name]["threshold"]
        bundle_path = save_detection_bundle(
            "artifacts/fraud_detection_bundle.joblib",
            model_name=deployable_name,
            model=anomaly_models[deployable_name],
            scaler=scaler,
            threshold=deployable_threshold,
            metadata={
                "dataset_source": source_info["source"],
                "is_synthetic": source_info["is_synthetic"],
                "split_strategy": split.strategy,
                "imbalance_strategy": imbalance_strategy,
                "false_negative_cost": FN_COST,
                "false_positive_cost": FP_COST,
                "validation_metrics": validation_table.loc[deployable_name].to_dict(),
                "test_metrics": test_table.loc[deployable_name].to_dict(),
            },
        )
        print("Saved deployment bundle:", bundle_path)
        print("Start API with: python app.py")
        """
    ),
    code(
        """
        # Example payload and local score; values come from a test observation.
        example_payload = split.test.iloc[0][["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]]
        example_frame = pd.DataFrame([example_payload.to_dict()])
        example_features = engineer_features(example_frame)[MODEL_FEATURES]
        example_score = anomaly_score(
            deployable_name,
            anomaly_models[deployable_name],
            scaler.transform(example_features),
        )[0]
        display({
            "anomaly_score": float(example_score),
            "threshold": float(deployable_threshold),
            "decision": "manual_review" if example_score >= deployable_threshold else "allow",
        })
        """
    ),
    markdown(
        """
        ## 9. Monitoring, governance, and retraining plan

        **Data controls**

        - Enforce schema, numeric ranges, null-rate limits, and feature order.
        - Track amount/time distributions and scoring-volume changes.
        - Keep personally identifiable information outside analytics logs.

        **Model controls**

        - Monitor alert rate, score distribution, precision, recall, PR-AUC, fraud loss captured,
          false-positive customer friction, and case-review turnaround time.
        - Calculate drift measures such as Population Stability Index when a stable reference
          window exists.
        - Revalidate the threshold when fraud cost, investigation capacity, or fraud patterns
          change. Retraining must never use future labels relative to the evaluation window.

        **Human oversight**

        - The model prioritizes transactions for review; it should not autonomously deny a
          customer without policy controls and appeal mechanisms.
        - Store model version, score, threshold, and reviewer outcome for auditability.
        - Use champion/challenger testing and rollback before replacing the production model.

        **Fairness limitation**

        Protected demographic attributes are absent because features are anonymized. Therefore,
        this benchmark cannot demonstrate demographic fairness. That absence is a limitation,
        not evidence that the model is fair.
        """
    ),
    markdown(
        """
        ## 10. Conclusions

        This project demonstrates that fraud detection is an operating-threshold problem as
        much as a modelling problem. The analysis compares global, local, statistical, and
        reconstruction-based detectors under a leakage-resistant evaluation design. It handles
        class imbalance through fit-for-purpose metrics, cost-sensitive thresholding, normal-only
        novelty training, and a SMOTE/class-weighted supervised reference.

        The recommended model must be taken from the executed comparison table rather than
        hard-coded in advance. A production rollout should begin as a decision-support control,
        collect reviewer feedback, monitor drift, and recalibrate costs and thresholds before
        any automated blocking action.

        ### Limitations and next steps

        - The benchmark covers only two days and anonymizes most features.
        - Fraud labels may arrive with delay in real operations.
        - Customer, merchant, device, geography, and graph relationships are unavailable.
        - Future work should test temporal cross-validation, calibrated supervised models,
          autoencoders, graph anomaly detection, explainability, and live drift monitoring.
        """
    ),
    markdown(
        """
        ## References

        1. Machine Learning Group, Universite Libre de Bruxelles. *Credit Card Fraud Detection* dataset. Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        2. AlmaBetter. *Mastering Advanced Machine Learning for Industry Applications* (course material): anomaly preprocessing, statistical/ML outlier detection, and fraud modelling.
        3. AlmaBetter. *Mastering ML Infrastructure from Model Training to Scalable Deployment* (course material): packaging, REST APIs, containerization, monitoring, and feedback loops.
        4. Scikit-learn documentation for Isolation Forest, Local Outlier Factor, PCA, Random Forest, and model-evaluation metrics: https://scikit-learn.org/stable/
        5. imbalanced-learn documentation for SMOTE and leakage-safe pipelines: https://imbalanced-learn.org/stable/
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "colab": {"name": "Financial_Transaction_Anomaly_Detection.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path("Financial_Transaction_Anomaly_Detection.ipynb")
output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {output} with {len(cells)} cells")
