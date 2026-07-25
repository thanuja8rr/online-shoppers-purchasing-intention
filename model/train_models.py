"""
BITS WILP - Machine Learning Assignment 2
Training script: Online Shoppers Purchasing Intention (UCI)

Trains five classification models on the same dataset and saves:
  - Preprocessor (ColumnTransformer)
  - Fitted models (.joblib)
  - Evaluation metrics (metrics.json)
  - Held-out test CSV for Streamlit upload demos
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
RAW_CSV = DATA_DIR / "online_shoppers_intention.csv"
TEST_CSV = ROOT / "test_data.csv"
METRICS_JSON = MODEL_DIR / "metrics.json"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_META_PATH = MODEL_DIR / "feature_meta.json"

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00468/online_shoppers_intention.csv"
)

TARGET_COL = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Explicit model registry (assignment-required classifiers)
MODEL_BUILDERS = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "kNN": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": lambda: GaussianNB(),
    "Random Forest (Ensemble)": lambda: RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}


def download_dataset() -> pd.DataFrame:
    """Load dataset from local cache or download from UCI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"Loading cached dataset: {RAW_CSV}")
        return pd.read_csv(RAW_CSV)

    print(f"Downloading dataset from UCI:\n  {DATASET_URL}")
    df = pd.read_csv(DATASET_URL)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved raw dataset -> {RAW_CSV}")
    return df


def prepare_xy(df: pd.DataFrame):
    """Split features/target and identify numeric vs categorical columns."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL]).copy()
    y_raw = df[TARGET_COL].copy()

    # Normalize boolean-like target to string labels for clarity
    y_raw = y_raw.map({True: "Purchase", False: "NoPurchase", 1: "Purchase", 0: "NoPurchase"})
    if y_raw.isna().any():
        # Fallback if already string labels
        y_raw = df[TARGET_COL].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Object/bool columns are categorical (Month, VisitorType, Weekend).
    # Integer-coded attributes (OS/Browser/Region/TrafficType) are kept numeric
    # as provided by UCI for stable scaling across all five model families.
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    meta = {
        "target": TARGET_COL,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "feature_order": list(X.columns),
        "class_names": list(label_encoder.classes_),
        "n_features": int(X.shape[1]),
        "n_instances": int(df.shape[0]),
        "dataset_name": "Online Shoppers Purchasing Intention (UCI)",
        "dataset_url": DATASET_URL,
    }
    return X, y, label_encoder, meta


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Scale numerics; one-hot encode categoricals (dense for GaussianNB)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def evaluate_model(model, X_test, y_test, class_names) -> dict:
    """Compute all assignment-required metrics on held-out test data."""
    y_pred = model.predict(X_test)

    # Probability / decision scores for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_score)),
        "Precision": float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = download_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    X, y, label_encoder, meta = prepare_xy(df)
    assert meta["n_features"] >= 12, "Dataset must have at least 12 features."
    assert meta["n_instances"] >= 500, "Dataset must have at least 500 instances."

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(meta["numeric_features"], meta["categorical_features"])
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Persist preprocessor + label encoder + metadata
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    with open(FEATURE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Export test set with human-readable target for Streamlit uploads
    test_export = X_test.copy()
    test_export[TARGET_COL] = label_encoder.inverse_transform(y_test)
    # Also keep a compact sample for repo if needed; full held-out set is fine (~2.4k rows)
    test_export.to_csv(TEST_CSV, index=False)
    print(f"Saved test data -> {TEST_CSV} ({len(test_export)} rows)")

    all_metrics = {}
    print("\nTraining & evaluating models...")
    print("-" * 72)

    for name, builder in MODEL_BUILDERS.items():
        model = builder()
        model.fit(X_train_t, y_train)
        metrics = evaluate_model(model, X_test_t, y_test, meta["class_names"])
        all_metrics[name] = metrics

        out_path = MODEL_DIR / MODEL_FILENAMES[name]
        joblib.dump(model, out_path)

        print(
            f"{name:28s} | Acc={metrics['Accuracy']:.4f} "
            f"AUC={metrics['AUC']:.4f} P={metrics['Precision']:.4f} "
            f"R={metrics['Recall']:.4f} F1={metrics['F1']:.4f} "
            f"MCC={metrics['MCC']:.4f}"
        )
        print(f"  saved -> {out_path.name}")

    # Persist metrics for README / Streamlit defaults
    serializable = {}
    for name, m in all_metrics.items():
        serializable[name] = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in m.items()
            if k in {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}
        }
        serializable[name]["confusion_matrix"] = m["confusion_matrix"]
        serializable[name]["classification_report"] = m["classification_report"]

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": meta,
                "split": {
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                },
                "metrics": serializable,
            },
            f,
            indent=2,
        )

    print("-" * 72)
    print(f"Metrics saved -> {METRICS_JSON}")
    print("Training complete.")


if __name__ == "__main__":
    main()
