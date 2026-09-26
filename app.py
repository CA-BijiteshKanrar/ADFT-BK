"""Minimal Flask scoring service for the selected anomaly detector."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from src.fraud_detection import MODEL_FEATURES, RAW_FEATURES, anomaly_score, engineer_features


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/fraud_detection_bundle.joblib"))
app = Flask(__name__)
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if bundle is not None else "model_not_loaded",
            "model_path": str(MODEL_PATH),
        }
    )


@app.post("/predict")
def predict():
    if bundle is None:
        return jsonify({"error": "Model artifact not found. Run the notebook first."}), 503

    payload = request.get_json(silent=True) or {}
    missing = [feature for feature in RAW_FEATURES if feature not in payload]
    if missing:
        return jsonify({"error": "Missing input fields", "fields": missing}), 400

    try:
        raw = pd.DataFrame([{key: float(payload[key]) for key in RAW_FEATURES}])
        engineered = engineer_features(raw)
        X = engineered[bundle.get("model_features", MODEL_FEATURES)].astype(float)
        X_scaled = bundle["scaler"].transform(X)
        score = float(
            anomaly_score(bundle["model_name"], bundle["model"], X_scaled)[0]
        )
        threshold = float(bundle["threshold"])
        return jsonify(
            {
                "is_suspicious": bool(score >= threshold),
                "anomaly_score": score,
                "threshold": threshold,
                "model": bundle["model_name"],
                "decision": "manual_review" if score >= threshold else "allow",
            }
        )
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
