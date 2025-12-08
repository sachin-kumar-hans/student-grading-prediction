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

from stacking_cv import load_dataset, build_stacking_model, train_full_model, predict_single


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(page_title="Student OUTPUT Grade – Stacking CV", layout="wide")

st.title("🎓 Student OUTPUT Grade Prediction – Stacking Ensemble")

st.markdown(
    """
This app can:

1. Run **k-fold stratified cross-validation** using a **stacking ensemble**  
   (Random Forest + Gradient Boosting + SVC → Logistic Regression as meta-learner)  
   to evaluate performance on predicting **OUTPUT Grade**.

2. Allow you to **enter student attributes manually** and get a predicted **OUTPUT Grade**.
"""
)

# -----------------------------
# 1. Load data from repo ONLY (no upload)
# -----------------------------
DATA_PATH = "data/final_data.csv"

try:
    df, _, _, _ = load_dataset(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Default dataset not found at `{DATA_PATH}`.\n\n"
        "Please ensure `data/final_data.csv` is present in your GitHub repository."
    )
    st.stop()

target_col = df.columns[-1]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.subheader("Dataset preview")
st.write(f"**Target column:** `{target_col}`")
st.write(f"**Samples:** {X.shape[0]}, **Features:** {X.shape[1]}")
st.dataframe(df.head())

st.markdown("---")

# -----------------------------
# 2. Mode selection
# -----------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Select mode",
    ("Cross-validation", "Single prediction (enter attributes)")
)

# ============================================================
# MODE 1: Cross-validation
# ============================================================
if mode == "Cross-validation":
    st.sidebar.header("Cross-Validation Settings")
    n_splits = st.sidebar.slider("Number of folds", min_value=3, max_value=10, value=5, step=1)
    show_plots = st.sidebar.checkbox("Show plots (Confusion Matrix & ROC per fold)", value=True)

    run_button = st.button("Run Cross-Validation")

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


# ============================================================
# MODE 2: Single prediction (manual input)
# ============================================================
elif mode == "Single prediction (enter attributes)":
    st.write("### Enter student attributes to predict OUTPUT Grade")
    st.info("Please enter whole-number codes as indicated for each attribute.")

    # Train model on full default data (no upload)
    model, feature_names, classes = train_full_model(DATA_PATH)

    # Help text / label mapping for each attribute
    feature_help = {
        "Student ID": "Student ID (e.g., 1, 2, 3)",
        "Student Age": "Student Age (1: 18-21, 2: 22-25, 3: above 26)",
        "Gender": "Gender (1: female, 2: male)",
        "Graduated high-school type": "Graduated high-school type (1: private, 2: state, 3: other)",
        "Scholarship type": "Scholarship type (1: None, 2: 25%, 3: 50%, 4: 75%, 5: Full)",
        "Additional work": "Additional work (1: Yes, 2: No)",
        "Regular artistic or sports activity": "Regular artistic or sports activity (1: Yes, 2: No)",
        "Do you have a partner": "Do you have a partner (1: Yes, 2: No)",
        "Total salary if available": "Total salary if available (1: USD 135-200, 2: USD 201-270, 3: USD 271-340, 4: USD 341-410, 5: above 410)",
        "Transportation to the university": "Transportation to the university (1: Bus, 2: Private car/taxi, 3: bicycle, 4: Other)",
        "Accommodation type in Cyprus": "Accommodation type in Cyprus (1: rental, 2: dormitory, 3: with family, 4: Other)",
        "Mothers' education": "Mother education (1: primary school, 2: secondary school, 3: high school, 4: university, 5: MSc., 6: Ph.D.)",
        "Fathers' education": "Father education (1: primary school, 2: secondary school, 3: high school, 4: university, 5: MSc., 6: Ph.D.)",
        "Number of sisters/brothers": "Number of sisters/brothers (1: 1, 2: 2, 3: 3, 4: 4, 5: 5 or above)",
        "Parental status": "Parental status (1: married, 2: divorced, 3: died - one of them or both)",
        "Mothers' occupation": "Mother occupation (1: retired, 2: housewife, 3: government officer, 4: private sector employee, 5: self-employment, 6: other)",
        "Fathers' occupation": "Father occupation (1: retired, 2: government officer, 3: private sector employee, 4: self-employment, 5: other)",
        "Weekly study hours": "Weekly study hours (1: None, 2: <5 hours, 3: 6-10 hours, 4: 11-20 hours, 5: more than 20 hours)",
        "Reading frequency (non-scientific books/journals)": "Reading frequency (non-scientific books/journals) (1: None, 2: Sometimes, 3: Often)",
        "Reading frequency (scientific books/journals)": "Reading frequency (scientific books/journals) (1: None, 2: Sometimes, 3: Often)",
        "Attendance to the seminars/conferences related to the department": "Attendance to the seminars/conferences related to the department (1: Yes, 2: No)",
        "Impact of your projects/activities on your success": "Impact of your projects/activities on your success (1: positive, 2: negative, 3: neutral)",
        "Attendance to classes": "Attendance to classes (1: always, 2: sometimes, 3: never)",
        "Preparation to midterm exams 1": "Preparation to midterm exams 1 (1: alone, 2: with friends, 3: not applicable)",
        "Preparation to midterm exams 2": "Preparation to midterm exams 2 (1: closest date to the exam, 2: regularly during the semester, 3: never)",
        "Taking notes in classes": "Taking notes in classes (1: never, 2: sometimes, 3: always)",
        "Listening in classes": "Listening in classes (1: never, 2: sometimes, 3: always)",
        "Discussion improves my interest and success in the course": "Discussion improves my interest and success in the course (1: never, 2: sometimes, 3: always)",
        "Flip-classroom": "Flip-classroom (1: not useful, 2: useful, 3: not applicable)",
        "Cumulative grade point average in the last semester (/4.00)": "Cumulative GPA last semester (/4.00) (1: <2.00, 2: 2.00-2.49, 3: 2.50-2.99, 4: 3.00-3.49, 5: above 3.49)",
        "Expected Cumulative grade point average in the graduation (/4.00)": "Expected cumulative GPA in graduation (/4.00) (1: <2.00, 2: 2.00-2.49, 3: 2.50-2.99, 4: 3.00-3.49, 5: above 3.49)",
        "Course ID": "Course ID (enter values between 0-9)"
    }

    st.write("Enter **integer values** for each attribute according to the codes given:")

    user_values = []
    cols = st.columns(2)  # two columns layout for cleaner UI

    for i, feat in enumerate(feature_names):
        # Skip target column if somehow included
        if feat == target_col:
            continue

        label = feature_help.get(feat, feat)
        col = cols[i % 2]
        with col:
            val = st.number_input(
                label=label,
                min_value=0,
                step=1,
                format="%d"
            )
        user_values.append(val)

    if st.button("Predict OUTPUT Grade"):
        pred_label, proba = predict_single(model, user_values)

        st.success(f"Predicted OUTPUT Grade: **{pred_label}**")

        st.write("### Class probabilities")
        proba_df = pd.DataFrame({
            "Class": classes,
            "Probability": proba
        })
        st.dataframe(proba_df.style.format({"Probability": "{:.4f}"}))
