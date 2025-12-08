import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

from stacking_cv import load_dataset, build_stacking_model


st.set_page_config(page_title="Student OUTPUT Grade – Stacking CV", layout="wide")

st.title("🎓 Student OUTPUT Grade Prediction – Stacking Ensemble (5-Fold CV)")

st.markdown(
    """
This app runs a **5-fold stratified cross-validation** using a **stacking ensemble**  
(Random Forest + Gradient Boosting + SVC → Logistic Regression as meta-learner)  
to predict the **OUTPUT Grade** of students.
"""
)

# 1. Load data (default or uploaded)
st.sidebar.header("Dataset Options")

uploaded_file = st.sidebar.file_uploader("Upload your CSV (optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded dataset.")
else:
    st.sidebar.info("Using default dataset: `data/final_data.csv`")
    df, _, _, _ = load_dataset("data/final_data.csv")

target_col = df.columns[-1]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.subheader("Dataset preview")
st.write(f"**Target column:** `{target_col}`")
st.write(f"**Samples:** {X.shape[0]}, **Features:** {X.shape[1]}")
st.dataframe(df.head())

st.markdown("---")

# 2. Controls
st.sidebar.header("Cross-Validation Settings")
n_splits = st.sidebar.slider("Number of folds", min_value=3, max_value=10, value=5, step=1)
show_plots = st.sidebar.checkbox("Show plots (Confusion Matrix & ROC per fold)", value=True)

run_button = st.button("Run Cross-Validation")

# 3. Run 5-fold CV when requested
if run_button:
    st.write("### Running Stratified Cross-Validation")
    classes = np.unique(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    roc_macro_scores = []

    fold_idx = 1

    for train_index, test_index in skf.split(X, y):
        st.markdown(f"## Fold {fold_idx}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Build a new stacking model for this fold
        model = build_stacking_model()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Multi-class ROC–AUC (macro, OvR)
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
        roc_macro_scores.append(roc_auc_macro)

        # Show metrics for this fold
        st.write(
            f"**Accuracy:** {acc:.4f}  |  "
            f"**Precision (macro):** {prec:.4f}  |  "
            f"**Recall (macro):** {rec:.4f}  |  "
            f"**F1-score (macro):** {f1:.4f}  |  "
            f"**ROC-AUC (macro OvR):** {roc_auc_macro:.4f}"
        )

        if show_plots:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title(f"Confusion Matrix - Fold {fold_idx}")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            st.pyplot(fig_cm)

            # ROC Curve (micro-averaged)
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            roc_auc_micro = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_micro:.3f}")
            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"Micro-Averaged ROC Curve - Fold {fold_idx}")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        fold_idx += 1

    
    # 4. Summary across folds
    st.markdown("---")
    st.write("## Cross-Validation Summary")

    summary_df = pd.DataFrame({
        "fold": list(range(1, n_splits + 1)),
        "accuracy": acc_scores,
        "precision_macro": prec_scores,
        "recall_macro": rec_scores,
        "f1_macro": f1_scores,
        "roc_auc_macro_ovr": roc_macro_scores
    })
    st.dataframe(summary_df.style.format("{:.4f}"))

    st.write("### Mean ± Std across folds")
    st.write(f"**Accuracy:** {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    st.write(f"**Precision (macro):** {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    st.write(f"**Recall (macro):** {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
    st.write(f"**F1-score (macro):** {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    st.write(f"**ROC-AUC (macro OvR):** {np.mean(roc_macro_scores):.4f} ± {np.std(roc_macro_scores):.4f}")
