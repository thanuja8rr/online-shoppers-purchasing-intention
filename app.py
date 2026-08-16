"""
BITS WILP - Machine Learning Assignment 2
Streamlit app: Online Shoppers Purchasing Intention classifiers

Features (assignment-required):
  a. Dataset upload (CSV test data)
  b. Model selection dropdown
  c. Evaluation metrics display
  d. Confusion matrix / classification report
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METRICS_JSON = MODEL_DIR / "metrics.json"
FEATURE_META_PATH = MODEL_DIR / "feature_meta.json"
DEFAULT_TEST_CSV = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

TARGET_COL = "Revenue"


@st.cache_resource
def load_artifacts():
    """Load preprocessor, label encoder, models, and cached metrics once."""
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(FEATURE_META_PATH, "r", encoding="utf-8") as f:
        feature_meta = json.load(f)
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics_bundle = json.load(f)

    models = {}
    for name, filename in MODEL_FILES.items():
        models[name] = joblib.load(MODEL_DIR / filename)
    return preprocessor, label_encoder, feature_meta, metrics_bundle, models


def normalize_target(series: pd.Series, class_names: list[str]) -> pd.Series:
    """Map common target encodings to the training label space."""
    mapping = {
        "true": "Purchase",
        "false": "NoPurchase",
        "1": "Purchase",
        "0": "NoPurchase",
        "yes": "Purchase",
        "no": "NoPurchase",
        "purchase": "Purchase",
        "nopurchase": "NoPurchase",
        "no purchase": "NoPurchase",
        "no_purchase": "NoPurchase",
    }
    cleaned = series.astype(str).str.strip()
    lowered = cleaned.str.lower()
    mapped = lowered.map(mapping).fillna(cleaned)

    # Keep values that already match class names; otherwise try title-case variants
    valid = set(class_names)
    out = mapped.apply(lambda v: v if v in valid else str(v))
    return out


def prepare_features(
    df: pd.DataFrame,
    feature_order: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Align uploaded CSV columns to the training feature order."""
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required feature columns: "
            + ", ".join(missing)
        )
    X = df[feature_order].copy()
    # Match training dtypes: categoricals were stored as strings
    for col in categorical_features or []:
        if col in X.columns:
            X[col] = X[col].astype(str)
    return X


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def style_app():
    st.set_page_config(
        page_title="Shopper Intent Classifiers | BITS WILP ML-A2",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Source Sans 3', sans-serif;
        }
        .main-title {
            font-size: 2.0rem;
            font-weight: 700;
            color: #0F3D3E;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #4A5C5C;
            font-size: 1.02rem;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background: linear-gradient(145deg, #F4FAFA 0%, #E8F3F1 100%);
            border: 1px solid #C5D9D6;
            border-radius: 10px;
            padding: 0.85rem 0.9rem;
            text-align: center;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #5B6E6E;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .metric-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.35rem;
            font-weight: 500;
            color: #0F3D3E;
        }
        .section-head {
            font-size: 1.15rem;
            font-weight: 700;
            color: #123C3D;
            border-left: 4px solid #2A9D8F;
            padding-left: 0.55rem;
            margin: 1.1rem 0 0.6rem 0;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F3D3E 0%, #1A5A5C 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #F2F7F7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict):
    keys = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    cols = st.columns(6)
    for col, key in zip(cols, keys):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value">{metrics[key]:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="BuGn", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def main():
    style_app()

    try:
        preprocessor, label_encoder, feature_meta, metrics_bundle, models = load_artifacts()
    except FileNotFoundError as exc:
        st.error(
            "Model artifacts not found. Run `python model/train_models.py` first "
            f"to train and save models.\n\nDetails: {exc}"
        )
        st.stop()

    class_names = feature_meta["class_names"]
    feature_order = feature_meta["feature_order"]

    st.markdown(
        '<div class="main-title">Online Shopper Purchase Intent — ML Classifier Suite</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">BITS WILP Machine Learning Assignment-2 · '
        "UCI Online Shoppers Purchasing Intention Dataset · "
        f"{feature_meta['n_features']} features · "
        f"{feature_meta['n_instances']} instances</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Controls")
        st.caption("Upload held-out test CSV and choose a trained model.")

        uploaded = st.file_uploader(
            "Upload test dataset (CSV)",
            type=["csv"],
            help="Upload only test data. Must include feature columns and optional Revenue target.",
        )
        use_default = st.checkbox(
            "Use bundled test_data.csv",
            value=uploaded is None,
            help="Loads the repository test split if no file is uploaded.",
        )

        model_name = st.selectbox(
            "Select classification model",
            options=list(MODEL_FILES.keys()),
            index=4,
        )

        st.markdown("---")
        st.markdown("**Dataset notes**")
        st.caption(
            "Binary target: Purchase vs NoPurchase. "
            "Numeric features are standardized; categoricals are one-hot encoded."
        )
        st.caption("Models were trained with a stratified 80/20 split (random_state=42).")

    # Resolve dataframe
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_source = f"Uploaded file: `{uploaded.name}`"
    elif use_default and DEFAULT_TEST_CSV.exists():
        df = pd.read_csv(DEFAULT_TEST_CSV)
        data_source = "Bundled `test_data.csv` (held-out test split)"
    else:
        st.warning("Please upload a CSV test file or enable the bundled test data option.")
        st.stop()

    st.markdown('<div class="section-head">1. Loaded Test Data</div>', unsafe_allow_html=True)
    st.write(data_source)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Selected Model", model_name)
    with st.expander("Preview first 10 rows"):
        st.dataframe(df.head(10), width="stretch")

    has_target = TARGET_COL in df.columns
    if not has_target:
        st.error(
            f"Column `{TARGET_COL}` is required in the uploaded CSV so evaluation "
            "metrics, confusion matrix, and classification report can be computed."
        )
        st.stop()

    try:
        X = prepare_features(
            df,
            feature_order,
            categorical_features=feature_meta.get("categorical_features", []),
        )
        y_labels = normalize_target(df[TARGET_COL], class_names)
        unknown = sorted(set(y_labels.unique()) - set(class_names))
        if unknown:
            st.error(
                f"Unrecognized target labels: {unknown}. "
                f"Expected one of: {class_names} (or True/False, 1/0)."
            )
            st.stop()
        y_true = label_encoder.transform(y_labels)
        X_t = preprocessor.transform(X)
    except Exception as exc:
        st.error(f"Failed to prepare uploaded data: {exc}")
        st.stop()

    model = models[model_name]
    y_pred = model.predict(X_t)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_t)[:, 1]
    else:
        y_score = y_pred.astype(float)

    live_metrics = compute_metrics(y_true, y_pred, y_score)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    st.markdown(
        '<div class="section-head">2. Evaluation Metrics (on uploaded / selected test data)</div>',
        unsafe_allow_html=True,
    )
    render_metric_cards(live_metrics)

    st.markdown(
        '<div class="section-head">3. Confusion Matrix & Classification Report</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1.2])
    with left:
        fig = plot_confusion_matrix(cm, class_names)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        report_df = pd.DataFrame(report).transpose()
        # Keep readable columns
        keep_cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns]
        st.dataframe(report_df[keep_cols].style.format(precision=4), width="stretch")
        st.caption("Classification report generated on the currently loaded test data.")

    st.markdown(
        '<div class="section-head">4. Cross-Model Comparison (benchmark test split)</div>',
        unsafe_allow_html=True,
    )
    cached = metrics_bundle.get("metrics", {})
    rows = []
    for name in MODEL_FILES:
        m = cached.get(name, {})
        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": m.get("Accuracy"),
                "AUC": m.get("AUC"),
                "Precision": m.get("Precision"),
                "Recall": m.get("Recall"),
                "F1": m.get("F1"),
                "MCC": m.get("MCC"),
            }
        )
    comparison_df = pd.DataFrame(rows)
    comparison_df.index = range(1, len(comparison_df) + 1)
    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ).background_gradient(cmap="YlGn", subset=["Accuracy", "AUC", "F1", "MCC"]),
        width="stretch",
    )

    with st.expander("Prediction sample (first 15 rows)"):
        preview = X.copy()
        preview["Actual"] = y_labels.values
        preview["Predicted"] = label_encoder.inverse_transform(y_pred)
        preview["Purchase_Probability"] = np.round(y_score, 4)
        st.dataframe(preview.head(15), width="stretch")

    st.markdown("---")
    st.caption(
        "BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning Assignment-2 · "
        "Interactive demonstration of five classical classifiers on the same UCI dataset."
    )


if __name__ == "__main__":
    main()
