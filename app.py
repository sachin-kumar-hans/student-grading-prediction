import numpy as np
import pandas as pd
import streamlit as st

from sklearn.exceptions import NotFittedError

from stacking_cv import load_dataset, train_full_model, predict_single

# -------------------------------------------------
# Basic config
# -------------------------------------------------
st.set_page_config(
    page_title="Student OUTPUT Grade – Prediction Demo",
    layout="wide"
)

st.title("🎓 Student OUTPUT Grade Prediction (Lightweight Demo)")

st.markdown(
    """
This lightweight app:

- Loads a prepared higher-education dataset from the repository  
- Trains a **stacking ensemble** once (cached)  
- Lets you enter student attributes as integer codes  
- Predicts the **OUTPUT Grade** for that student  

All heavy cross-validation and plots have been removed so it runs smoothly on Streamlit Cloud.
"""
)

# -------------------------------------------------
# 1. Load data from repo (no upload)
# -------------------------------------------------
DATA_PATH = "data/final_data.csv"

try:
    df, X, y, target_col = load_dataset(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Default dataset not found at `{DATA_PATH}`.\n\n"
        "Please ensure `data/final_data.csv` exists in your GitHub repository."
    )
    st.stop()
except pd.errors.EmptyDataError:
    st.error(
        f"`{DATA_PATH}` exists but is **empty**.\n\n"
        "Open this file in GitHub and paste your full CSV contents into it, "
        "then save (commit) the file."
    )
    st.stop()

st.subheader("Dataset preview")
st.write(f"**Target column:** `{target_col}`")
st.write(f"**Samples:** {X.shape[0]}, **Features:** {X.shape[1]}")
st.dataframe(df.head())

st.markdown("---")

# -------------------------------------------------
# 2. Cache trained model (very important for Cloud)
# -------------------------------------------------
@st.cache_resource
def get_trained_model_and_metadata():
    """
    Train stacking model on full dataset and cache it.
    Returns: model, feature_names, classes
    """
    model, feature_names, classes = train_full_model(DATA_PATH)
    return model, feature_names, classes

try:
    model, feature_names, classes = get_trained_model_and_metadata()
except Exception as e:
    st.error(f"Error while training the model: {e}")
    st.stop()

# -------------------------------------------------
# 3. Helper: labels with coding instructions
# -------------------------------------------------
feature_help = {
    "Student ID": "Student ID (any whole number, e.g., 1, 2, 3)",
    "Student Age": "Student Age (1: 18–21, 2: 22–25, 3: above 26)",
    "Gender": "Gender (1: female, 2: male)",
    "Sex": "Gender / Sex (1: female, 2: male)",
    "Graduated high-school type": "Graduated high-school type (1: private, 2: state, 3: other)",
    "Scholarship type": "Scholarship type (1: None, 2: 25%, 3: 50%, 4: 75%, 5: Full)",
    "Additional work": "Additional work (1: Yes, 2: No)",
    "Regular artistic or sports activity": "Regular artistic or sports activity (1: Yes, 2: No)",
    "Do you have a partner": "Do you have a partner (1: Yes, 2: No)",
    "Total salary if available": (
        "Total salary if available "
        "(1: USD 135–200, 2: 201–270, 3: 271–340, 4: 341–410, 5: above 410)"
    ),
    "Transportation to the university": (
        "Transportation to the university "
        "(1: Bus, 2: Private car/taxi, 3: bicycle, 4: Other)"
    ),
    "Accommodation type in Cyprus": (
        "Accommodation type in Cyprus "
        "(1: rental, 2: dormitory, 3: with family, 4: Other)"
    ),
    "Mothers' education": (
        "Mother education (1: primary school, 2: secondary school, 3: high school, "
        "4: university, 5: MSc., 6: Ph.D.)"
    ),
    "Fathers' education": (
        "Father education (1: primary school, 2: secondary school, 3: high school, "
        "4: university, 5: MSc., 6: Ph.D.)"
    ),
    "Number of sisters/brothers": (
        "Number of sisters/brothers (1: 1, 2: 2, 3: 3, 4: 4, 5: 5 or above)"
    ),
    "Parental status": "Parental status (1: married, 2: divorced, 3: died - one of them or both)",
    "Mothers' occupation": (
        "Mother occupation (1: retired, 2: housewife, 3: government officer, "
        "4: private sector employee, 5: self-employment, 6: other)"
    ),
    "Fathers' occupation": (
        "Father occupation (1: retired, 2: government officer, 3: private sector employee, "
        "4: self-employment, 5: other)"
    ),
    "Weekly study hours": (
        "Weekly study hours (1: None, 2: <5 hours, 3: 6–10 hours, "
        "4: 11–20 hours, 5: more than 20 hours)"
    ),
    "Reading frequency (non-scientific books/journals)": (
        "Reading frequency (non-scientific books/journals) "
        "(1: None, 2: Sometimes, 3: Often)"
    ),
    "Reading frequency (scientific books/journals)": (
        "Reading frequency (scientific books/journals) "
        "(1: None, 2: Sometimes, 3: Often)"
    ),
    "Attendance to the seminars/conferences related to the department": (
        "Attendance to department seminars/conferences (1: Yes, 2: No)"
    ),
    "Impact of your projects/activities on your success": (
        "Impact of your projects/activities on your success "
        "(1: positive, 2: negative, 3: neutral)"
    ),
    "Attendance to classes": "Attendance to classes (1: always, 2: sometimes, 3: never)",
    "Preparation to midterm exams 1": (
        "Preparation to midterm exams 1 (1: alone, 2: with friends, 3: not applicable)"
    ),
    "Preparation to midterm exams 2": (
        "Preparation to midterm exams 2 "
        "(1: closest date to the exam, 2: regularly during the semester, 3: never)"
    ),
    "Taking notes in classes": "Taking notes in classes (1: never, 2: sometimes, 3: always)",
    "Listening in classes": "Listening in classes (1: never, 2: sometimes, 3: always)",
    "Discussion improves my interest and success in the course": (
        "Discussion improves my interest and success in the course "
        "(1: never, 2: sometimes, 3: always)"
    ),
    "Flip-classroom": "Flip-classroom (1: not useful, 2: useful, 3: not applicable)",
    "Cumulative grade point average in the last semester (/4.00)": (
        "Cumulative GPA last semester (/4.00) "
        "(1: <2.00, 2: 2.00–2.49, 3: 2.50–2.99, 4: 3.00–3.49, 5: above 3.49)"
    ),
    "Expected Cumulative grade point average in the graduation (/4.00)": (
        "Expected cumulative GPA in graduation (/4.00) "
        "(1: <2.00, 2: 2.00–2.49, 3: 2.50–2.99, 4: 3.00–3.49, 5: above 3.49)"
    ),
    "Course ID": "Course ID (any whole number, e.g., 1–9)"
}

# -------------------------------------------------
# 4. Input form for single prediction
# -------------------------------------------------
st.markdown("## 🔢 Predict OUTPUT Grade for a New Student")

st.info(
    "Please enter **integer codes** for each attribute as described. "
    "All fields are required."
)

user_values = []
cols = st.columns(2)  # two-column layout

for i, feat in enumerate(feature_names):
    # feature_names should already exclude the target column
    label = feature_help.get(feat, feat)  # fall back to raw name if not in dict
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
    try:
        pred_label, proba = predict_single(model, user_values)

        st.success(f"🎯 Predicted OUTPUT Grade: **{pred_label}**")

        st.markdown("### Class probabilities")
        proba_df = pd.DataFrame({
            "Class": classes,
            "Probability": proba
        })
        st.dataframe(proba_df.style.format({"Probability": "{:.4f}"}))

    except NotFittedError:
        st.error("Model is not fitted. Please retrain or check the training step.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
