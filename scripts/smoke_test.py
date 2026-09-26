"""Fast executable check that does not require the real dataset."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.preprocessing import RobustScaler

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


def main() -> None:
    raw = generate_synthetic_transactions(n_samples=12_000, fraud_ratio=0.01)
    clean, _ = clean_transactions(raw)
    frame = engineer_features(clean)
    split = chronological_split(frame)
    scaler = RobustScaler().fit(split.train[MODEL_FEATURES])
    normal_train = split.train.loc[split.train["Class"] == 0, MODEL_FEATURES]
    models = fit_anomaly_models(scaler.transform(normal_train), fast_mode=True)

    summary = {}
    for name, model in models.items():
        validation_scores = anomaly_score(
            name, model, scaler.transform(split.validation[MODEL_FEATURES])
        )
        choice = select_cost_threshold(split.validation["Class"], validation_scores)
        test_scores = anomaly_score(
            name, model, scaler.transform(split.test[MODEL_FEATURES])
        )
        summary[name] = classification_metrics(
            split.test["Class"], test_scores, choice["threshold"]
        )

    output = Path("reports/smoke_test_metrics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

