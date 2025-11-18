import os
import streamlit as st
import joblib
import numpy as np

# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(ROOT, "data", "features")
MODELS_DIR   = os.path.join(ROOT, "data", "models")

VECT_PATH = os.path.join(FEATURES_DIR, "tfidf_vectorizer.joblib")


# ============================================================
# HELPERS
# ============================================================

def get_latest_model():
    """Return path to the newest ticket_classifier_*.joblib file."""
    candidates = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("ticket_classifier_") and f.endswith(".joblib")
    ]

    if not candidates:
        raise RuntimeError("No trained models found in /data/models")

    return os.path.join(MODELS_DIR, max(candidates))


@st.cache_resource
def load_model_and_vectorizer():
    """Load model + TF-IDF vectorizer and cache them."""
    model_path = get_latest_model()
    vectorizer = joblib.load(VECT_PATH)
    model = joblib.load(model_path)
    return model, vectorizer


def predict(model, vectorizer, text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Cummins Ticket Classifier",
    page_icon="🛠️",
    layout="centered"
)

st.title("🛠️ Cummins IT Ticket Classifier")
st.write("Paste any IT ticket description and this AI will classify it automatically.")

model, vectorizer = load_model_and_vectorizer()

# Text input area
ticket_text = st.text_area(
    "Enter ticket description:",
    placeholder="Example: 'Cannot log into SAP, receiving user authentication failure...'",
    height=200
)

if st.button("Classify Ticket"):
    if len(ticket_text.strip()) < 5:
        st.warning("Please enter a longer ticket description.")
    else:
        pred, proba = predict(model, vectorizer, ticket_text)

        st.subheader("🔍 Prediction")
        st.success(f"**Category:** {pred}")

        # Show probability distribution
        st.subheader("📊 Model Confidence")
        labels = model.classes_
        conf_df = { "Category": labels, "Probability": np.round(proba, 4) }

        st.bar_chart(
            data={"Probability": proba},
            x=labels,
            y="Probability"
        )

        st.write("Done.")
