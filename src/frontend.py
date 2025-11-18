import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import pipeline

# -----------------------
# Paths
# -----------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(ROOT, "data", "features")
MODELS_DIR = os.path.join(ROOT, "data", "models")
VECT_PATH = os.path.join(FEATURES_DIR, "tfidf_vectorizer.joblib")

# -----------------------
# Sub-label definitions
# -----------------------
SUB_LABELS = {
    "Access": [
        "password reset", "account locked", "VPN access",
        "shared drive access", "email access", "permission denied",
    ],
    "Administrative rights": [
        "local admin rights", "software installation rights",
        "run as administrator", "elevated permissions",
    ],
    "HR Support": [
        "onboarding", "offboarding", "HR system access",
        "payroll issue", "holiday and leave",
        "employee self-service portal",
    ],
    "Hardware": [
        "laptop issue", "desktop issue", "monitor problem",
        "keyboard or mouse", "printer issue", "docking station",
    ],
    "Internal Project": [
        "project access", "project tooling", "test environment",
        "release deployment",
    ],
    "Miscellaneous": [
        "general enquiry", "investigation required",
        "other uncategorised issue",
    ],
    "Purchase": [
        "new laptop request", "new software request",
        "licence purchase", "hardware procurement",
    ],
    "Storage": [
        "quota exceeded", "archive request", "shared folder storage",
        "cloud storage issue", "backup and restore",
    ],
}

# -----------------------
# Load newest trained model
# -----------------------
def get_latest_model():
    files = [f for f in os.listdir(MODELS_DIR)
             if f.startswith("ticket_classifier_") and f.endswith(".joblib")]
    if not files:
        raise RuntimeError("No trained models found.")
    return os.path.join(MODELS_DIR, max(files))

# -----------------------
# Load model + vectorizer + HF
# -----------------------
@st.cache_resource
def load_components():
    model_path = get_latest_model()
    vectorizer = joblib.load(VECT_PATH)
    model = joblib.load(model_path)

    hf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    return model, vectorizer, hf

# -----------------------
# Main label prediction
# -----------------------
def predict_main(model, vectorizer, text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        scores = model.decision_function(X)[0]
        proba = softmax(scores)

    return pred, proba

# -----------------------
# Sub-label prediction via HF
# -----------------------
def predict_sub_label(text, main_label, hf):
    candidates = SUB_LABELS.get(main_label)
    if not candidates:
        return None, None

    out = hf(text, candidate_labels=candidates, multi_label=True)
    return out["labels"][0], out["scores"][0]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Ticket Classifier")

st.title("IT Ticket Classifier")
st.write("Paste any IT ticket description and the AI will classify it into a main category and a sub-category.")

model, vectorizer, hf_clf = load_components()

ticket_text = st.text_area(
    "Enter ticket description:",
    placeholder="Example: 'Cannot log into SAP, receiving user authentication failure...'",
    height=200
)

if st.button("Classify Ticket"):
    if len(ticket_text.strip()) < 5:
        st.warning("Please enter a longer ticket description.")
    else:
        # main label
        main_label, proba = predict_main(model, vectorizer, ticket_text)

        # sub label
        sub_label, sub_score = predict_sub_label(ticket_text, main_label, hf_clf)

        # -------------------
        # Display results
        # -------------------
        st.subheader("Prediction")
        st.success(f"**Main Category:** {main_label}")

        if sub_label:
            st.info(f"**Sub-Category:** {sub_label} (score ≈ {sub_score:.2f})")

        # Probability chart
        df = pd.DataFrame({
            "Category": model.classes_,
            "Probability": np.round(proba, 4)
        })

        st.subheader("Model Confidence")
        st.bar_chart(df.set_index("Category"))