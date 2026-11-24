import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer ML Demo", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "breast_cancer_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    medians = json.loads((MODEL_DIR / "feature_medians.json").read_text())
    return model, scaler, feature_names, medians

model, scaler, feature_names, medians = load_artifacts()

st.title("🩺 Breast Cancer Classification — ML Demonstration")
st.write(
    "This educational application demonstrates a machine-learning model "
    "trained on the Wisconsin Diagnostic Breast Cancer dataset."
)
st.error(
    "Research/education only. This tool is not a medical device, is not "
    "clinically validated, and must not be used to diagnose, rule out, or "
    "guide treatment for breast cancer."
)
st.info(
    "The model expects 30 numeric measurements extracted from digitized "
    "breast-mass cell nuclei. It does not accept mammograms, X-rays, or photos."
)

with st.expander("What do the input measurements mean?"):
    st.write(
        "The dataset contains mean, standard-error (_se), and worst "
        "measurements for 10 cell-nuclei characteristics. Use only values "
        "that match the dataset's feature definitions and units. The default "
        "values below are training-set medians and are placeholders, not "
        "patient measurements."
    )

with st.form("measurements"):
    values = {}
    st.subheader("Enter the 30 numerical features")
    groups = [
        ("Mean measurements", [f for f in feature_names if f.endswith("_mean")]),
        ("Standard-error measurements", [f for f in feature_names if f.endswith("_se")]),
        ("Worst measurements", [f for f in feature_names if f.endswith("_worst")]),
    ]
    for group_name, group_features in groups:
        st.markdown(f"**{group_name}**")
        cols = st.columns(3)
        for i, feature in enumerate(group_features):
            with cols[i % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=float(medians[feature]),
                    format="%.6f",
                    key=feature,
                    help="Default is the training-set median, not an individual patient's measurement."
                )
    submitted = st.form_submit_button("Run model prediction", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
    scaled = scaler.transform(row)
    prediction = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]
    class_to_index = {int(c): i for i, c in enumerate(model.classes_)}
    benign_prob = float(probabilities[class_to_index[0]])
    malignant_prob = float(probabilities[class_to_index[1]])

    st.subheader("Model output")
    if prediction == 1:
        st.warning("Model classification: Malignant (class 1)")
    else:
        st.success("Model classification: Benign (class 0)")

    c1, c2 = st.columns(2)
    c1.metric("Benign model score", f"{benign_prob:.1%}")
    c2.metric("Malignant model score", f"{malignant_prob:.1%}")
    st.caption(
        "These are model probability outputs, not calibrated clinical risk "
        "estimates. A prediction from this demo must not be interpreted as a diagnosis."
    )

st.divider()
st.caption("Dataset: Wisconsin Diagnostic Breast Cancer (WDBC). Model: Logistic Regression.")
