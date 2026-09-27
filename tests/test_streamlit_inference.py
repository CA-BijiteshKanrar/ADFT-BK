"""Integration checks for the two deployable artifacts and their raw input contract."""

import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_scoring import RAW_FEATURES, score_transactions


ROOT = Path(__file__).resolve().parents[1]


class StreamlitInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = pd.read_csv(ROOT / "samples" / "example_transactions.csv")
        cls.models = [
            joblib.load(ROOT / "models" / "random_forest.joblib"),
            joblib.load(ROOT / "models" / "anomaly_detector.joblib"),
        ]

    def test_both_models_score_target_free_examples(self):
        self.assertEqual(list(self.examples.columns), RAW_FEATURES)
        for bundle in self.models:
            with self.subTest(model=bundle["model_name"]):
                predictions = score_transactions(bundle, self.examples)
                self.assertEqual(len(predictions), len(self.examples))
                self.assertTrue(np.isfinite(predictions["score"]).all())
                self.assertEqual(predictions.iloc[1]["decision"], "manual_review")

    def test_extra_label_does_not_change_scores(self):
        labelled = self.examples.assign(Class=[0, 1, 0])
        for bundle in self.models:
            with self.subTest(model=bundle["model_name"]):
                plain = score_transactions(bundle, self.examples)
                with_label = score_transactions(bundle, labelled)
                np.testing.assert_allclose(plain["score"], with_label["score"], rtol=1e-12, atol=1e-12)

    def test_invalid_csv_fields_are_rejected(self):
        invalid_rows = [
            self.examples.drop(columns=["V1"]),
            self.examples.assign(Amount=-1.0),
            self.examples.assign(V2=float("nan")),
        ]
        for frame in invalid_rows:
            for bundle in self.models:
                with self.subTest(model=bundle["model_name"], columns=list(frame.columns)):
                    with self.assertRaises(ValueError):
                        score_transactions(bundle, frame)


if __name__ == "__main__":
    unittest.main()
