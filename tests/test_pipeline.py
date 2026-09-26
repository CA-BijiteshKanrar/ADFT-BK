from __future__ import annotations

import numpy as np

from src.fraud_detection import (
    MODEL_FEATURES,
    anomaly_score,
    chronological_split,
    classification_metrics,
    clean_transactions,
    engineer_features,
    fit_anomaly_models,
    generate_synthetic_transactions,
    select_cost_threshold,
)


def test_end_to_end_smoke():
    raw = generate_synthetic_transactions(n_samples=5_000, fraud_ratio=0.015)
    clean, audit = clean_transactions(raw)
    assert audit["rows_after"] == len(clean)

    featured = engineer_features(clean)
    assert set(MODEL_FEATURES).issubset(featured.columns)
    split = chronological_split(featured)

    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit(split.train[MODEL_FEATURES])
    normal = split.train.loc[split.train["Class"] == 0, MODEL_FEATURES]
    models = fit_anomaly_models(scaler.transform(normal), fast_mode=True)
    model = models["Isolation Forest"]

    validation_scores = anomaly_score(
        "Isolation Forest",
        model,
        scaler.transform(split.validation[MODEL_FEATURES]),
    )
    selection = select_cost_threshold(split.validation["Class"], validation_scores)
    test_scores = anomaly_score(
        "Isolation Forest",
        model,
        scaler.transform(split.test[MODEL_FEATURES]),
    )
    metrics = classification_metrics(
        split.test["Class"], test_scores, selection["threshold"]
    )
    assert 0 <= metrics["Precision"] <= 1
    assert 0 <= metrics["Recall"] <= 1
    assert np.isfinite(metrics["Threshold"])

