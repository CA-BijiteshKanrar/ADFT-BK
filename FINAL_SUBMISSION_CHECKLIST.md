# Final Submission Checklist

## Real-data execution

- [ ] Download the ULB `creditcard.csv` dataset.
- [ ] Place it in `data/creditcard.csv` or `/content/creditcard.csv`.
- [ ] Keep `ALLOW_SYNTHETIC_SMOKE_TEST = False`.
- [ ] Set `FAST_MODE = False`.
- [ ] Run every notebook cell from a clean runtime.
- [ ] Confirm `source_info["is_synthetic"]` is `False`.
- [ ] Confirm the test comparison, curves, and confusion matrix render correctly.
- [ ] Confirm `reports/model_comparison_test.csv` and
  `artifacts/fraud_detection_bundle.joblib` are created.

## Report and presentation

- [ ] Copy the real executed metrics into the Results section of
  `PROJECT_REPORT.md`, if a separate written report is requested.
- [ ] Replace every `[RUN OUTPUT: ...]` marker in
  `VIDEO_PRESENTATION_SCRIPT.md`.
- [ ] Record the presentation within 25 minutes.
- [ ] Show the notebook outputs and briefly demonstrate the API or saved model.
- [ ] Do not claim the model is production-ready or demographically fair.

## GitHub repository

- [ ] Push all project files except `creditcard.csv` and generated model files.
- [ ] Confirm the README renders and reproduction commands are correct.
- [ ] Add the executed notebook or a link to the executed Colab notebook.
- [ ] Test the repository from a fresh environment.
- [ ] Make the repository accessible to the evaluator.

## Notebook sharing

- [ ] Share with view access for everyone who has the link.
- [ ] Grant edit access to `evaluator@almabetter.com`.
- [ ] Open the shared link in a private/incognito window to confirm access.

## Academic integrity

- [ ] Retain dataset and library attribution.
- [ ] Describe any assistance received in accordance with course policy.
- [ ] Ensure the explanation reflects your own understanding.
- [ ] Verify that all reported metrics came from the real dataset run.

