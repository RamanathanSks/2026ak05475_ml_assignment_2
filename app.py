import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ml_models import (  # metric helpers (NumPy only); the models themselves are loaded from model/*.joblib
    confusion, accuracy, auc, precision, recall, f1, mcc, classification_report_df,
)

st.set_page_config(page_title="Diagnostic Classifier Evaluation Suite", layout="wide")

st.title("Diagnostic ML Classifier Evaluation Suite")
st.markdown("**Name:** Ramanathan Sks &nbsp;&nbsp;|&nbsp;&nbsp; **BITS ID:** 2026ak05475")
st.markdown(
    "Upload the test split (`test_data.csv`) and choose a pre-trained model to inspect live evaluation metrics, "
    "confusion matrix, and class-wise performance. Models are loaded from `model/` and **never retrained**."
)

MODEL_REGISTRY = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
LABELS = ["Malignant (0)", "Benign (1)"]


@st.cache_resource(show_spinner="Loading model...")
def _load(path, mtime):
    return joblib.load(path)


def load_model(name):
    path = MODEL_REGISTRY[name]
    return _load(path, os.path.getmtime(path))  # reload automatically if the file is re-saved


with st.sidebar:
    st.header("Configuration")
    uploaded_csv = st.file_uploader("1. Upload Test Dataset (CSV)", type=["csv"])
    chosen_model = st.selectbox("2. Select Classification Model", list(MODEL_REGISTRY.keys()))

if uploaded_csv is not None:
    test_df = pd.read_csv(uploaded_csv)

    if "target" not in test_df.columns:
        st.error("Error: Uploaded CSV must contain the 'target' column.")
    else:
        X_eval = test_df.drop(columns=["target"]).to_numpy(float)
        y_eval = test_df["target"].to_numpy(int)

        clf = load_model(chosen_model)          # pre-trained: load only, no fit()
        y_hat = clf.predict(X_eval)
        y_prob = clf.predict_proba(X_eval)

        st.subheader(f"Performance Metrics — {chosen_model}")
        cols = st.columns(6)
        cols[0].metric("Accuracy", f"{accuracy(y_eval, y_hat):.4f}")
        cols[1].metric("AUC Score", f"{auc(y_eval, y_prob):.4f}")
        cols[2].metric("Precision", f"{precision(y_eval, y_hat):.4f}")
        cols[3].metric("Recall", f"{recall(y_eval, y_hat):.4f}")
        cols[4].metric("F1 Score", f"{f1(y_eval, y_hat):.4f}")
        cols[5].metric("MCC Score", f"{mcc(y_eval, y_hat):.4f}")

        st.divider()
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Confusion Matrix")
            tp, tn, fp, fn = confusion(y_eval, y_hat)
            cm = np.array([[tn, fp], [fn, tp]])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="YlGnBu")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
            ax.set_xticks([0, 1], LABELS)
            ax.set_yticks([0, 1], LABELS)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with right_col:
            st.subheader("Classification Report")
            st.dataframe(classification_report_df(y_eval, y_hat).style.format("{:.4f}"))
else:
    st.info("Please upload `test_data.csv` from the sidebar to evaluate the models.")
