"""Interactive transaction scoring for the course project's fitted model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from fraud_scoring import RAW_FEATURES, score_transactions


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Supervised Random Forest": ROOT / "models" / "random_forest.joblib",
    "Validation-selected anomaly detector": ROOT / "models" / "anomaly_detector.joblib",
}
SAMPLE_PATH = ROOT / "samples" / "example_transactions.csv"
MAX_BATCH_ROWS = 10_000


@st.cache_resource
def load_models() -> dict[str, dict]:
    """Read the trusted, repository-bundled artifacts once per app process."""
    expected = {"model", "model_kind", "model_name", "model_features", "threshold", "metadata"}
    models = {}
    for label, path in MODEL_PATHS.items():
        bundle = joblib.load(path)
        if not expected.issubset(bundle):
            raise ValueError(f"The {label} artifact is incomplete.")
        models[label] = bundle
    return models


@st.cache_data
def load_examples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def render_single(bundle: dict, examples: pd.DataFrame) -> None:
    st.subheader("Score one transaction")
    st.write("Start with a sample transaction, then adjust the raw fields. The `Class` label is not used for scoring.")
    chosen = st.selectbox("Example", range(len(examples)), format_func=lambda i: f"Example {i + 1}")
    sample = examples.iloc[chosen]

    with st.form("single_transaction"):
        left, right = st.columns(2)
        with left:
            time = st.number_input("Time · seconds since the first transaction", min_value=0.0, value=float(sample["Time"]), format="%.2f", key=f"time_{chosen}")
        with right:
            amount = st.number_input("Amount", min_value=0.0, value=float(sample["Amount"]), format="%.2f", key=f"amount_{chosen}")
        values = {"Time": time, "Amount": amount}
        with st.expander("Anonymized features V1–V28", expanded=False):
            columns = st.columns(4)
            for i in range(1, 29):
                name = f"V{i}"
                with columns[(i - 1) % 4]:
                    values[name] = st.number_input(name, value=float(sample[name]), format="%.6f", key=f"{chosen}_{name}")
        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        prediction = score_transactions(bundle, pd.DataFrame([values]))
        score = float(prediction.iloc[0]["score"])
        decision = prediction.iloc[0]["decision"]
        label = "Fraud probability" if bundle["model_kind"] == "supervised_probability" else "Anomaly score"
        st.metric(label, f"{score:.4f}")
        st.caption(f"Review threshold: {bundle['threshold']:.4f}")
        if decision == "manual_review":
            st.warning("Flagged for manual review. This is not proof of fraud.")
        else:
            st.success("Below the review threshold.")


def render_batch(bundle: dict, examples: pd.DataFrame) -> None:
    st.subheader("Score a CSV batch")
    st.write("Provide `Time`, `Amount`, and `V1` through `V28`. Extra columns, including `Class`, are ignored. Up to 10,000 rows are accepted.")
    st.download_button(
        "Download sample CSV",
        data=examples.loc[:, RAW_FEATURES].to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload transaction CSV", type="csv")
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded, nrows=MAX_BATCH_ROWS + 1)
        if len(frame) > MAX_BATCH_ROWS:
            raise ValueError(f"Upload no more than {MAX_BATCH_ROWS:,} rows at a time.")
        result = score_transactions(bundle, frame)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Cannot score this file: {exc}")
        return

    review_count = int((result["decision"] == "manual_review").sum())
    first, second = st.columns(2)
    first.metric("Transactions scored", f"{len(result):,}")
    second.metric("Sent to manual review", f"{review_count:,}")
    output = pd.concat([frame.loc[:, RAW_FEATURES], result], axis=1)
    st.dataframe(output, width="stretch", hide_index=True)
    st.download_button(
        "Download scored CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Financial Transaction Review", page_icon="💳", layout="wide")
    st.title("Financial Transaction Review")
    st.caption("Course-project demonstration using the anonymized ULB/Worldline credit-card dataset")

    try:
        models = load_models()
        examples = load_examples()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"App files are unavailable or invalid: {exc}")
        st.stop()

    with st.sidebar:
        st.subheader("Choose a fitted model")
        selected = st.selectbox("Scoring model", list(models))
        bundle = models[selected]
        st.write(bundle["model_name"])
        score_type = "Fraud probability" if bundle["model_kind"] == "supervised_probability" else "Anomaly score"
        st.write(f"**Score type:** {score_type}")
        st.write(f"**Review threshold:** {bundle['threshold']:.4f}")
        metadata = bundle["metadata"]
        st.write(f"**Training rows:** {metadata.get('training_rows', '—'):,}")
        st.write("Threshold selected on validation data; test data were held out.")
        st.link_button("View project notebook", "https://colab.research.google.com/github/CA-BijiteshKanrar/ADFT-BK/blob/main/Financial_Transaction_Anomaly_Detection.ipynb")

    batch, single, details = st.tabs(["CSV Import", "Single transaction", "Model comparison"])
    with batch:
        render_batch(bundle, examples)
    with single:
        render_single(bundle, examples)
    with details:
        st.markdown(bundle["metadata"].get("model_note", ""))
        st.write("A flag routes a transaction for review. Do not use this public demo with real customer data or to make payment-blocking decisions.")
        st.write("**Locked test-set snapshot**")
        comparison = pd.DataFrame(
            {
                label: item["metadata"]["test_metrics"]
                for label, item in models.items()
            }
        ).T
        st.dataframe(comparison.round(4), width="stretch")
        st.caption("Both models used the same chronological split. Their thresholds were chosen on validation data. The anomaly grid for this app was narrowed for repeatable deployment training; the comparison describes these packaged artifacts.")
        st.write("The benchmark spans only two days and omits customer, merchant, device, and geographic context. Its offline results do not establish live performance.")


if __name__ == "__main__":
    main()
