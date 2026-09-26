"""Core utilities for the financial transaction anomaly-detection project.

The module is intentionally framework-light so the same preprocessing and
scoring code can be used from the notebook, tests, and Flask service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


V_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_FEATURES, "Amount"]
REQUIRED_COLUMNS = [*RAW_FEATURES, "Class"]
MODEL_FEATURES = [*V_FEATURES, "LogAmount", "HourSin", "HourCos"]


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str


class PCAReconstructionDetector:
    """PCA detector whose anomaly score is scaled reconstruction error."""

    def __init__(self, n_components: float | int = 0.95, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca_: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAReconstructionDetector":
        self.pca_ = PCA(
            n_components=self.n_components,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("The detector must be fitted before scoring.")
        reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        return np.mean(np.square(X - reconstructed), axis=1)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    required = REQUIRED_COLUMNS if require_target else RAW_FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if require_target:
        labels = set(pd.to_numeric(df["Class"], errors="coerce").dropna().unique())
        if not labels.issubset({0, 1}):
            raise ValueError("Class must contain only binary labels 0 and 1.")


def generate_synthetic_transactions(
    n_samples: int = 30_000,
    fraud_ratio: float = 0.006,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a schema-compatible dataset for development smoke tests only."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=12,
        n_redundant=8,
        n_repeated=0,
        n_clusters_per_class=3,
        weights=[1.0 - fraud_ratio, fraud_ratio],
        class_sep=1.45,
        flip_y=0.0005,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(X, columns=V_FEATURES)
    frame["Time"] = np.sort(rng.uniform(0, 172_800, size=n_samples))
    normal_amount = rng.lognormal(mean=3.1, sigma=1.0, size=n_samples)
    fraud_multiplier = np.where(y == 1, rng.uniform(1.1, 3.0, n_samples), 1.0)
    frame["Amount"] = np.round(normal_amount * fraud_multiplier, 2)
    frame["Class"] = y.astype(int)
    return frame[["Time", *V_FEATURES, "Amount", "Class"]]


def load_transactions(
    csv_path: str | Path = "data/creditcard.csv",
    *,
    allow_openml: bool = True,
    allow_synthetic: bool = False,
    synthetic_rows: int = 30_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the ULB dataset locally, then OpenML, then an explicit smoke fallback."""
    candidates = [Path(csv_path), Path("/content/creditcard.csv")]
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            validate_schema(df)
            return df, {
                "source": str(candidate.resolve()),
                "is_synthetic": False,
                "note": "Loaded from local CSV.",
            }

    openml_error: str | None = None
    if allow_openml:
        try:
            bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
            df = bunch.frame.copy()
            if "Class" not in df.columns and bunch.target is not None:
                df["Class"] = bunch.target
            df["Class"] = pd.to_numeric(df["Class"], errors="raise").astype(int)
            validate_schema(df)
            return df, {
                "source": "OpenML dataset 1597 (creditcard)",
                "is_synthetic": False,
                "note": "Downloaded through sklearn.fetch_openml.",
            }
        except Exception as exc:  # network/auth varies by execution platform
            openml_error = f"{exc.__class__.__name__}: {exc}"

    if allow_synthetic:
        df = generate_synthetic_transactions(
            n_samples=synthetic_rows,
            random_state=random_state,
        )
        return df, {
            "source": "Synthetic development data",
            "is_synthetic": True,
            "note": (
                "SMOKE TEST ONLY. Replace with the ULB creditcard.csv before "
                "submitting results."
            ),
            "openml_error": openml_error,
        }

    raise FileNotFoundError(
        "creditcard.csv was not found and OpenML could not be loaded. Download "
        "the ULB dataset to data/creditcard.csv. OpenML error: "
        f"{openml_error or 'not attempted'}"
    )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Coerce types, remove exact duplicates, and document data-quality actions."""
    validate_schema(df)
    clean = df[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    missing_before = clean.isna().sum().to_dict()
    rows_before = len(clean)
    clean = clean.dropna().drop_duplicates().reset_index(drop=True)
    clean["Class"] = clean["Class"].astype(int)
    clean["Amount"] = clean["Amount"].clip(lower=0)
    validate_schema(clean)
    audit = {
        "rows_before": rows_before,
        "rows_after": len(clean),
        "rows_removed": rows_before - len(clean),
        "missing_before": {k: int(v) for k, v in missing_before.items()},
    }
    return clean, audit


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust amount and cyclical time features without using the label."""
    validate_schema(df, require_target=False)
    out = df.copy()
    seconds_in_day = np.mod(out["Time"].to_numpy(dtype=float), 86_400.0)
    hour = seconds_in_day / 3_600.0
    out["LogAmount"] = np.log1p(out["Amount"].clip(lower=0))
    out["HourSin"] = np.sin(2 * np.pi * hour / 24.0)
    out["HourCos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> SplitData:
    """Use deployment-like time splits, with a documented stratified fallback."""
    from sklearn.model_selection import train_test_split

    ordered = df.sort_values("Time").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if all(part["Class"].nunique() == 2 for part in (train, validation, test)):
        return SplitData(train, validation, test, "chronological 70/15/15")

    train, remainder = train_test_split(
        ordered,
        test_size=1 - train_fraction,
        stratify=ordered["Class"],
        random_state=random_state,
    )
    relative_test = (1 - train_fraction - validation_fraction) / (1 - train_fraction)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test,
        stratify=remainder["Class"],
        random_state=random_state,
    )
    return SplitData(
        train.sort_values("Time").copy(),
        validation.sort_values("Time").copy(),
        test.sort_values("Time").copy(),
        "stratified fallback because a chronological partition lacked a class",
    )


def cap_rows(
    X: np.ndarray,
    max_rows: int,
    random_state: int = 42,
) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(random_state)
    return X[rng.choice(len(X), size=max_rows, replace=False)]


def fit_anomaly_models(
    X_train_normal_scaled: np.ndarray,
    *,
    fast_mode: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit complementary global, local, and reconstruction-based detectors."""
    iso_rows = 25_000 if fast_mode else 100_000
    lof_rows = 10_000 if fast_mode else 45_000
    normal_iso = cap_rows(X_train_normal_scaled, iso_rows, random_state)
    normal_lof = cap_rows(X_train_normal_scaled, lof_rows, random_state)

    models: dict[str, Any] = {
        "Isolation Forest": IsolationForest(
            n_estimators=200 if fast_mode else 350,
            max_samples="auto",
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        ).fit(normal_iso),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=35,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        ).fit(normal_lof),
        "PCA Reconstruction": PCAReconstructionDetector(
            n_components=0.95,
            random_state=random_state,
        ).fit(normal_iso),
    }
    return models


def tune_anomaly_models(
    X_train_normal_scaled: np.ndarray,
    X_validation_scaled: np.ndarray,
    y_validation: np.ndarray,
    *,
    false_negative_cost: float = 25.0,
    false_positive_cost: float = 1.0,
    fast_mode: bool = False,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune compact, explainable grids on validation cost within each family."""
    iso_rows = 25_000 if fast_mode else 100_000
    lof_rows = 10_000 if fast_mode else 35_000
    normal_iso = cap_rows(X_train_normal_scaled, iso_rows, random_state)
    normal_lof = cap_rows(X_train_normal_scaled, lof_rows, random_state)

    iso_grid = (
        [(120, "auto")]
        if fast_mode
        else [(200, "auto"), (350, "auto"), (350, 0.50), (350, 0.80)]
    )
    lof_grid = [20] if fast_mode else [20, 35, 50]
    pca_grid = [0.95] if fast_mode else [0.90, 0.95, 0.99]

    candidates: list[tuple[str, dict[str, Any], Any, np.ndarray]] = []
    for n_estimators, max_samples in iso_grid:
        params = {"n_estimators": n_estimators, "max_samples": max_samples}
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        ).fit(normal_iso)
        candidates.append(("Isolation Forest", params, model, X_validation_scaled))

    for n_neighbors in lof_grid:
        params = {"n_neighbors": n_neighbors}
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        ).fit(normal_lof)
        candidates.append(("Local Outlier Factor", params, model, X_validation_scaled))

    for n_components in pca_grid:
        params = {"variance_retained": n_components}
        model = PCAReconstructionDetector(
            n_components=n_components,
            random_state=random_state,
        ).fit(normal_iso)
        candidates.append(("PCA Reconstruction", params, model, X_validation_scaled))

    rows: list[dict[str, Any]] = []
    fitted: list[tuple[str, Any]] = []
    for family, params, model, X_val in candidates:
        scores = anomaly_score(family, model, X_val)
        choice = select_cost_threshold(
            y_validation,
            scores,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        metrics = classification_metrics(y_validation, scores, choice["threshold"])
        rows.append(
            {
                "Family": family,
                "Parameters": json.dumps(params, sort_keys=True),
                "Validation_Cost": choice["validation_cost"],
                "F2": metrics["F2"],
                "PR_AUC": metrics["PR_AUC"],
                "Recall": metrics["Recall"],
                "Precision": metrics["Precision"],
                "Threshold": choice["threshold"],
            }
        )
        fitted.append((family, model))

    tuning_table = pd.DataFrame(rows)
    tuning_table["Selected"] = False
    selected: dict[str, Any] = {}
    for family in tuning_table["Family"].unique():
        family_rows = tuning_table.loc[tuning_table["Family"] == family]
        best_index = family_rows.sort_values(
            ["Validation_Cost", "F2", "PR_AUC"],
            ascending=[True, False, False],
        ).index[0]
        selected[family] = fitted[int(best_index)][1]
        tuning_table.loc[best_index, "Selected"] = True
    return selected, tuning_table.sort_values(
        ["Family", "Selected", "Validation_Cost"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def anomaly_score(model_name: str, model: Any, X_scaled: np.ndarray) -> np.ndarray:
    """Return scores where larger always means more anomalous."""
    raw = np.asarray(model.score_samples(X_scaled), dtype=float)
    if model_name == "PCA Reconstruction":
        return raw
    return -raw


def iqr_amount_score(df: pd.DataFrame, train_reference: pd.DataFrame) -> np.ndarray:
    """A transparent univariate statistical baseline using log transaction amount."""
    train_log = np.log1p(train_reference["Amount"].clip(lower=0))
    q1, q3 = train_log.quantile([0.25, 0.75])
    iqr = max(float(q3 - q1), 1e-9)
    center = float(train_log.median())
    return np.abs(np.log1p(df["Amount"].clip(lower=0)).to_numpy() - center) / iqr


def select_cost_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    false_negative_cost: float = 25.0,
    false_positive_cost: float = 1.0,
) -> dict[str, float]:
    """Choose a validation-only threshold minimizing explicit business cost."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision = precision[:-1]
    recall = recall[:-1]
    positives = max(int(y.sum()), 1)
    tp = recall * positives
    predicted_positive = np.divide(
        tp,
        precision,
        out=np.full_like(tp, fill_value=float(len(y))),
        where=precision > 0,
    )
    fp = np.maximum(predicted_positive - tp, 0)
    fn = positives - tp
    costs = false_negative_cost * fn + false_positive_cost * fp
    best = int(np.nanargmin(costs))
    return {
        "threshold": float(thresholds[best]),
        "validation_cost": float(costs[best]),
        "false_negative_cost": float(false_negative_cost),
        "false_positive_cost": float(false_positive_cost),
    }


def classification_metrics(
    y_true: Iterable[int],
    scores: Iterable[float],
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR_AUC": float(average_precision_score(y, s)),
        "ROC_AUC": float(roc_auc_score(y, s)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "F2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "False_Positive_Rate": float(fp / max(fp + tn, 1)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": float(threshold),
    }


def precision_at_k(y_true: Iterable[int], scores: Iterable[float], k: int) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    k = min(max(int(k), 1), len(y))
    indices = np.argpartition(s, -k)[-k:]
    return float(y[indices].mean())


def fit_supervised_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    fast_mode: bool = False,
) -> tuple[Any, str]:
    """Fit SMOTE+RF when available; otherwise use balanced class weights."""
    rf = RandomForestClassifier(
        n_estimators=180 if fast_mode else 400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline

        model = Pipeline(
            steps=[
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=0.05,
                        k_neighbors=5,
                        random_state=random_state,
                    ),
                ),
                ("random_forest", rf),
            ]
        )
        model.fit(X_train, y_train)
        return model, "SMOTE (training only) + class-weighted Random Forest"
    except ImportError:
        rf.fit(X_train, y_train)
        return rf, "Class-weighted Random Forest (imbalanced-learn unavailable)"


def save_detection_bundle(
    output_path: str | Path,
    *,
    model_name: str,
    model: Any,
    scaler: RobustScaler,
    threshold: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "threshold": float(threshold),
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "metadata": metadata or {},
    }
    joblib.dump(bundle, output)
    return output


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
