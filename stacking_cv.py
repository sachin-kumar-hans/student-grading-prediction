
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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


# -------------------------------------------------
# 1. Dataset loading
# -------------------------------------------------
def load_dataset(path: str):
    """
    Load dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
    X  : pd.DataFrame
        Feature matrix (all columns except last).
    y  : pd.Series
        Target vector (last column).
    target_col : str
        Name of the target column.
    """
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return df, X, y, target_col


# -------------------------------------------------
# 2. Stacking model builder
# -------------------------------------------------
def build_stacking_model():
    """
    Create a fresh stacking classifier with:
    - RandomForest
    - GradientBoosting
    - SVC (with scaling)
    and LogisticRegression as meta-learner.
    """

    # Base 1: Random Forest
    base_rf = ("rf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))

    # Base 2: Gradient Boosting
    base_gb = ("gb", GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))

    # Base 3: SVC inside a Pipeline with scaling
    base_svc = (
        "svc",
        Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                probability=True,
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=42
            ))
        ])
    )

    # Meta-learner: Logistic Regression (internally optimized via LBFGS)
    meta_clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )

    model = StackingClassifier(
        estimators=[base_rf, base_gb, base_svc],
        final_estimator=meta_clf,
        stack_method="auto",
        passthrough=False,
        n_jobs=-1
    )

    return model


# -------------------------------------------------
# 3. 5-Fold Stratified Cross-Validation
# -------------------------------------------------
def run_5fold_cv(X, y, plot: bool = True):
    """
    Run 5-fold Stratified CV with the stacking model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Labels.
    plot : bool, default=True
        If True, plots confusion matrix and ROC curve for each fold.

    Returns
    -------
    results : dict
        Dictionary with per-fold scores and summary statistics.
        Keys: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc_macro'
    """

    classes = np.unique(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    roc_macro_scores = []

    fold_idx = 1

    for train_index, test_index in skf.split(X, y):
        print(f"\n========== Fold {fold_idx} ==========")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Build a new stacking model for this fold
        model = build_stacking_model()

        # Fit model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics for this fold
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Multi-class ROC–AUC
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
        roc_macro_scores.append(roc_auc_macro)

        print(f"Fold {fold_idx} - Accuracy : {acc:.4f}")
        print(f"Fold {fold_idx} - Precision: {prec:.4f}")
        print(f"Fold {fold_idx} - Recall   : {rec:.4f}")
        print(f"Fold {fold_idx} - F1-score : {f1:.4f}")
        print(f"Fold {fold_idx} - ROC-AUC (macro, OvR): {roc_auc_macro:.4f}")

        if plot:
            # Confusion Matrix for this fold
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - Fold {fold_idx}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()

            # ROC Curve (Micro-averaged) for this fold
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            roc_auc_micro = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"Fold {fold_idx} (AUC = {roc_auc_micro:.3f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Micro-Averaged ROC Curve - Fold {fold_idx}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()

        fold_idx += 1

    # Summary
    print("\n========== Cross-Validation Summary (5 folds) ==========")
    print(f"Mean Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Mean Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    print(f"Mean Recall   : {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
    print(f"Mean F1-score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Mean ROC-AUC (macro, OvR): {np.mean(roc_macro_scores):.4f} ± {np.std(roc_macro_scores):.4f}")

    results = {
        "accuracy": np.array(acc_scores),
        "precision": np.array(prec_scores),
        "recall": np.array(rec_scores),
        "f1": np.array(f1_scores),
        "roc_auc_macro": np.array(roc_macro_scores),
    }

    return results


# -------------------------------------------------
# 4. Script entry point (optional)
# -------------------------------------------------
if __name__ == "__main__":
    # You can change this path or make it relative like "data/final_data.csv"
    data_path = r"E:\25. GDC Lecture\Higher Education UCI Data\final data.csv"

    df, X, y, target_col = load_dataset(data_path)
    print("Full data shape:", X.shape, "Target shape:", y.shape)
    print("Target column:", target_col)

    results = run_5fold_cv(X, y, plot=True)

    # Example: print means from returned results
    print("\nFrom returned results dict:")
    for metric_name, values in results.items():
        print(
            f"{metric_name}: mean={values.mean():.4f} ± {values.std():.4f}"
        )
