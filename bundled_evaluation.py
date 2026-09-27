"""Run the packaged models on the same held-out rows used during training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score

from fraud_scoring import RAW_FEATURES, score_transactions


DATA_PATH = Path(__file__).resolve().parent / "content" / "creditcard.csv.zip"


def load_held_out_test() -> tuple[pd.DataFrame, dict]:
    """Recreate the training script's cleaning and chronological test split."""
    required = [*RAW_FEATURES, "Class"]
    raw = pd.read_csv(DATA_PATH, compression="infer", usecols=required)
    clean = raw.loc[:, required].copy()
    for name in required:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
    clean = clean.dropna().drop_duplicates().reset_index(drop=True)
    clean["Class"] = clean["Class"].astype(int)
    clean["Amount"] = clean["Amount"].clip(lower=0)
    if not set(clean["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")

    ordered = clean.sort_values("Time").reset_index(drop=True)
    train_end = int(0.70 * len(ordered))
    validation_end = int(0.85 * len(ordered))
    test = ordered.iloc[validation_end:].copy()
    # A changed dataset could trigger the training script's stratified fallback.
    if not all(part["Class"].nunique() == 2 for part in (ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], test)):
        raise ValueError("The bundled dataset no longer supports the saved chronological split.")
    source = {
        "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "test_rows": len(test),
        "test_frauds": int(test["Class"].sum()),
    }
    return test, source


def evaluate_held_out_test(bundle: dict, test: pd.DataFrame, source: dict) -> tuple[pd.DataFrame, dict]:
    """Score held-out rows and calculate metrics from their known labels."""
    metadata = bundle["metadata"]
    for key in ("sha256", "test_rows", "test_frauds"):
        if source[key] != metadata[key]:
            raise ValueError(f"The bundled dataset does not match the fitted model ({key}).")

    predictions = score_transactions(bundle, test.loc[:, RAW_FEATURES])
    labels = test["Class"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    predicted = (predictions["decision"] == "manual_review").to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    report = {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "F2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "PR_AUC": float(average_precision_score(labels, scores)),
        "ROC_AUC": float(roc_auc_score(labels, scores)),
    }
    output = test.loc[:, ["Time", "Amount", "Class"]].reset_index(drop=True)
    output.insert(0, "test_row", np.arange(1, len(output) + 1))
    output["score"] = scores
    output["decision"] = predictions["decision"].to_numpy()
    return output, report
