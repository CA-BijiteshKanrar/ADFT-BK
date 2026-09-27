"""Feature preparation and inference shared by the Streamlit app and training script."""

from __future__ import annotations

import numpy as np
import pandas as pd


V_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_FEATURES, "Amount"]
MODEL_FEATURES = [*V_FEATURES, "LogAmount", "HourSin", "HourCos"]


def prepare_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order raw, target-free transaction columns."""
    missing = [name for name in RAW_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Add at least one transaction row.")

    try:
        raw = frame.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All transaction fields must contain numeric values.") from exc
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Transaction fields cannot be blank, NaN, or infinite.")
    if (raw["Time"] < 0).any() or (raw["Amount"] < 0).any():
        raise ValueError("Time and Amount must be zero or greater.")
    return raw.astype(float)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same log amount and circular time encoding as the notebook."""
    transactions = prepare_transactions(raw)
    hours = np.mod(transactions["Time"].to_numpy(), 86_400.0) / 3_600.0
    features = transactions.loc[:, V_FEATURES].copy()
    features["LogAmount"] = np.log1p(transactions["Amount"].to_numpy())
    features["HourSin"] = np.sin(2 * np.pi * hours / 24.0)
    features["HourCos"] = np.cos(2 * np.pi * hours / 24.0)
    return features.loc[:, MODEL_FEATURES]


def score_transactions(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Return review decisions without requiring Class labels at inference time."""
    raw = prepare_transactions(frame)
    features = engineer_features(raw).loc[:, bundle["model_features"]]
    model_kind = bundle["model_kind"]
    if model_kind == "supervised_probability":
        scores = bundle["model"].predict_proba(features.to_numpy())[:, 1]
    elif model_kind == "pca_reconstruction":
        scaled = bundle["scaler"].transform(features)
        reconstructed = bundle["model"].inverse_transform(
            bundle["model"].transform(scaled)
        )
        scores = np.mean(np.square(scaled - reconstructed), axis=1)
    elif model_kind == "novelty_score":
        scaled = bundle["scaler"].transform(features)
        scores = -bundle["model"].score_samples(scaled)
    elif model_kind == "iqr_amount":
        reference = bundle["reference"]
        scores = np.abs(np.log1p(raw["Amount"].to_numpy()) - reference["median"]) / reference["iqr"]
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    threshold = float(bundle["threshold"])
    return pd.DataFrame(
        {
            "score": scores,
            "decision": np.where(scores >= threshold, "manual_review", "allow"),
        },
        index=frame.index,
    )
