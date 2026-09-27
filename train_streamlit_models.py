"""Rebuild the two pre-fitted artifacts served by the Streamlit app.

Run from the repository root with the packages in requirements.txt installed.
The validation and test protocol mirrors the Colab notebook. Training happens
offline; the public app only loads the resulting trusted artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from fraud_scoring import MODEL_FEATURES, RAW_FEATURES, engineer_features


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "content" / "creditcard.csv.zip"
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "samples"
SEED = 42
FN_COST = 25.0
FP_COST = 1.0


def load_clean_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_PATH, compression="infer")
    required = [*RAW_FEATURES, "Class"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing {missing}")

    clean = raw.loc[:, required].copy()
    for name in required:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
    clean = clean.dropna().drop_duplicates().reset_index(drop=True)
    clean["Class"] = clean["Class"].astype(int)
    clean["Amount"] = clean["Amount"].clip(lower=0)
    if not set(clean["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")

    features = engineer_features(clean)
    full = pd.concat([clean, features.loc[:, ["LogAmount", "HourSin", "HourCos"]]], axis=1)
    ordered = full.sort_values("Time").reset_index(drop=True)
    train_end = int(0.70 * len(ordered))
    validation_end = int(0.85 * len(ordered))
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    split_strategy = "chronological 70/15/15"
    if not all(part["Class"].nunique() == 2 for part in (train, validation, test)):
        train, remainder = train_test_split(
            ordered, test_size=0.30, stratify=ordered["Class"], random_state=SEED
        )
        validation, test = train_test_split(
            remainder, test_size=0.50, stratify=remainder["Class"], random_state=SEED
        )
        split_strategy = "stratified fallback because a chronological partition lacked a class"

    source = {
        "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "training_frauds": int(train["Class"].sum()),
        "validation_frauds": int(validation["Class"].sum()),
        "test_frauds": int(test["Class"].sum()),
        "split_strategy": split_strategy,
        "sklearn_version": sklearn.__version__,
    }
    return train, validation, test, source


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Replicate the notebook's validation-only 25:1 threshold choice."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    positives = max(int(labels.sum()), 1)
    tp = recall * positives
    predicted_positive = np.divide(
        tp,
        precision,
        out=np.full_like(tp, fill_value=float(len(labels))),
        where=precision > 0,
    )
    fp = np.maximum(predicted_positive - tp, 0)
    fn = positives - tp
    cost = FN_COST * fn + FP_COST * fp
    best = int(np.nanargmin(cost))
    return float(thresholds[best]), float(cost[best])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "F2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "PR_AUC": float(average_precision_score(labels, scores)),
        "ROC_AUC": float(roc_auc_score(labels, scores)),
    }


def capped_rows(matrix: np.ndarray, limit: int) -> np.ndarray:
    if len(matrix) <= limit:
        return matrix
    choices = np.random.default_rng(SEED).choice(len(matrix), size=limit, replace=False)
    return matrix[choices]


def pca_scores(model: PCA, scaled: np.ndarray) -> np.ndarray:
    reconstruction = model.inverse_transform(model.transform(scaled))
    return np.mean(np.square(scaled - reconstruction), axis=1)


def amount_reference(train: pd.DataFrame) -> dict:
    log_amount = np.log1p(train["Amount"].clip(lower=0))
    q1, q3 = log_amount.quantile([0.25, 0.75])
    return {"median": float(log_amount.median()), "iqr": max(float(q3 - q1), 1e-9)}


def amount_scores(frame: pd.DataFrame, reference: dict) -> np.ndarray:
    values = np.log1p(frame["Amount"].clip(lower=0)).to_numpy()
    return np.abs(values - reference["median"]) / reference["iqr"]


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 1_000_000:.1f} MB)")


def train_supervised(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, source: dict) -> None:
    x_train = train[MODEL_FEATURES].to_numpy(dtype=float)
    y_train = train["Class"].to_numpy(dtype=int)
    x_validation = validation[MODEL_FEATURES].to_numpy(dtype=float)
    y_validation = validation["Class"].to_numpy(dtype=int)
    x_test = test[MODEL_FEATURES].to_numpy(dtype=float)
    y_test = test["Class"].to_numpy(dtype=int)

    # This matches the notebook's SMOTE and RF settings; only training is resampled.
    model = Pipeline(
        steps=[
            ("smote", SMOTE(sampling_strategy=0.05, k_neighbors=3, random_state=SEED)),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold, cost = select_threshold(y_validation, validation_scores)
    test_scores = model.predict_proba(x_test)[:, 1]
    bundle = {
        "model_kind": "supervised_probability",
        "model_name": "SMOTE + class-weighted Random Forest",
        "model": model,
        "scaler": None,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "threshold": threshold,
        "metadata": {
            **source,
            "validation_cost": cost,
            "validation_metrics": metrics(y_validation, validation_scores, threshold),
            "test_metrics": metrics(y_test, test_scores, threshold),
            "model_note": (
                "**Supervised benchmark.** This model was trained on labelled normal and fraud transactions. "
                "SMOTE with `k_neighbors=3` is applied inside the training pipeline only, followed by "
                "a class-weighted Random Forest. Its score is a model fraud probability, but it has not "
                "been calibrated for current banking traffic. The alert threshold was selected on "
                "validation data using an illustrative 25:1 missed-fraud to false-alert cost ratio."
            ),
        },
    }
    save_bundle(bundle, MODEL_DIR / "random_forest.joblib")
    print("Random Forest test metrics:", json.dumps(bundle["metadata"]["test_metrics"], indent=2))


def train_anomaly(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, source: dict) -> None:
    scaler = RobustScaler()
    x_train = scaler.fit_transform(train[MODEL_FEATURES])
    x_validation = scaler.transform(validation[MODEL_FEATURES])
    x_test = scaler.transform(test[MODEL_FEATURES])
    y_train = train["Class"].to_numpy(dtype=int)
    y_validation = validation["Class"].to_numpy(dtype=int)
    y_test = test["Class"].to_numpy(dtype=int)
    x_normal = x_train[y_train == 0]
    normal_iso = capped_rows(x_normal, 100_000)
    normal_lof = capped_rows(x_normal, 35_000)

    # Deployment retraining uses a smaller, resource-conscious validation grid.
    candidates: list[tuple[str, dict, object, object]] = []
    for trees, samples in [(200, "auto"), (350, "auto")]:
        model = IsolationForest(
            n_estimators=trees,
            max_samples=samples,
            contamination="auto",
            random_state=SEED,
            n_jobs=-1,
        ).fit(normal_iso)
        candidates.append(("Isolation Forest", {"n_estimators": trees, "max_samples": samples}, model, -model.score_samples(x_validation)))
    for neighbours in [20, 35, 50]:
        model = LocalOutlierFactor(
            n_neighbors=neighbours,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        ).fit(normal_lof)
        candidates.append(("Local Outlier Factor", {"n_neighbors": neighbours}, model, -model.score_samples(x_validation)))
    for variance in [0.90, 0.95, 0.99]:
        model = PCA(n_components=variance, svd_solver="full", random_state=SEED).fit(normal_iso)
        candidates.append(("PCA Reconstruction", {"variance_retained": variance}, model, pca_scores(model, x_validation)))

    selected: dict[str, dict] = {}
    for family, params, model, scores in candidates:
        threshold, cost = select_threshold(y_validation, scores)
        values = metrics(y_validation, scores, threshold)
        record = {"family": family, "params": params, "model": model, "scores": scores, "threshold": threshold, "cost": cost, "metrics": values}
        previous = selected.get(family)
        rank = (cost, -values["F2"], -values["PR_AUC"])
        if previous is None or rank < (previous["cost"], -previous["metrics"]["F2"], -previous["metrics"]["PR_AUC"]):
            selected[family] = record
        print(f"{family} {params}: cost={cost:.1f}, F2={values['F2']:.4f}", flush=True)

    reference = amount_reference(train)
    baseline_scores = amount_scores(validation, reference)
    baseline_threshold, baseline_cost = select_threshold(y_validation, baseline_scores)
    selected["IQR Amount Baseline"] = {
        "family": "IQR Amount Baseline",
        "params": reference,
        "model": None,
        "scores": baseline_scores,
        "threshold": baseline_threshold,
        "cost": baseline_cost,
        "metrics": metrics(y_validation, baseline_scores, baseline_threshold),
    }
    # The notebook's anomaly-only choice maximizes validation F2 across family winners.
    winner = max(selected.values(), key=lambda record: record["metrics"]["F2"])
    name = winner["family"]
    model_kind = {
        "Isolation Forest": "novelty_score",
        "Local Outlier Factor": "novelty_score",
        "PCA Reconstruction": "pca_reconstruction",
        "IQR Amount Baseline": "iqr_amount",
    }[name]
    if model_kind == "pca_reconstruction":
        test_scores = pca_scores(winner["model"], x_test)
    elif model_kind == "iqr_amount":
        test_scores = amount_scores(test, reference)
    else:
        test_scores = -winner["model"].score_samples(x_test)

    bundle = {
        "model_kind": model_kind,
        "model_name": name,
        "model": winner["model"],
        "scaler": None if model_kind == "iqr_amount" else scaler,
        "reference": reference if model_kind == "iqr_amount" else None,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "threshold": winner["threshold"],
        "metadata": {
            **source,
            "selected_parameters": winner["params"],
            "validation_cost": winner["cost"],
            "validation_metrics": winner["metrics"],
            "test_metrics": metrics(y_test, test_scores, winner["threshold"]),
            "model_note": (
                f"**Anomaly-only selection: {name}.** Each detector was fitted on normal training "
                "transactions. A compact deployment grid was used. The selected setting within each family minimized validation cost; "
                "the family winner was chosen by validation F2. Its score measures unusualness, "
                "not a calibrated fraud probability. A later fraud label is still needed to measure "
                "performance. The threshold uses an illustrative 25:1 cost ratio."
            ),
        },
    }
    save_bundle(bundle, MODEL_DIR / "anomaly_detector.joblib")
    print("Selected anomaly detector:", name, winner["params"])
    print("Anomaly test metrics:", json.dumps(bundle["metadata"]["test_metrics"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rf", "anomaly", "both"], default="both")
    args = parser.parse_args()
    train, validation, test, source = load_clean_split()
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    examples = pd.concat(
        [test.iloc[[0]], test.loc[test["Class"] == 1].iloc[[0]], test.iloc[[100]]],
        ignore_index=True,
    )
    examples.loc[:, RAW_FEATURES].to_csv(SAMPLE_DIR / "example_transactions.csv", index=False)
    print("Dataset and split:", json.dumps(source, indent=2), flush=True)
    if args.model in {"rf", "both"}:
        train_supervised(train, validation, test, source)
    if args.model in {"anomaly", "both"}:
        train_anomaly(train, validation, test, source)


if __name__ == "__main__":
    main()
