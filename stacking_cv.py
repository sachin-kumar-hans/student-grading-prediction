# stacking_cv.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def load_dataset(path: str):
    """
    Load dataset from CSV.

    Assumes:
    - Last column is the target (OUTPUT Grade).
    - All previous columns are numeric features.

    Returns:
        df          : full DataFrame
        X           : features (DataFrame)
        y           : target (Series)
        target_col  : target column name
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column and 1 target column.")

    target_col = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return df, X, y, target_col


def build_stacking_model():
    """
    Build the stacking model:
    - Base learners: RF, GB, SVC(with scaling)
    - Meta-learner: Logistic Regression
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

    # Meta-learner: Logistic Regression
    meta_clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )

    stack_clf = StackingClassifier(
        estimators=[base_rf, base_gb, base_svc],
        final_estimator=meta_clf,
        stack_method="auto",
        passthrough=False,
        n_jobs=-1
    )

    return stack_clf


def train_full_model(csv_path: str = "data/final_data.csv"):
    """
    Train stacking model on the full dataset.

    Returns:
        model          : trained stacking model
        feature_names  : list of feature column names (in order)
        classes        : array of target class labels (model.classes_)
    """
    df, X, y, target_col = load_dataset(csv_path)

    model = build_stacking_model()
    model.fit(X, y)

    classes = model.classes_

    return model, list(X.columns), classes


def predict_single(model, feature_values):
    """
    Predict OUTPUT Grade for a single sample.

    Parameters
    ----------
    model          : trained stacking model
    feature_values : list or 1D array of feature values (len = n_features)

    Returns
    -------
    pred_label     : predicted class label (e.g., 0,1,2,...,7)
    pred_proba     : probability of the predicted class (float in [0,1])
    """
    feature_values = np.array(feature_values, dtype=float).reshape(1, -1)

    # Probability distribution over classes
    proba_vec = model.predict_proba(feature_values)[0]

    # Index of the most likely class
    best_idx = int(np.argmax(proba_vec))

    # Corresponding label and probability
    pred_label = model.classes_[best_idx]
    pred_proba = float(proba_vec[best_idx])

    return pred_label, pred_proba
