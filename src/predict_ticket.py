import os
import numpy as np
import joblib
import pandas as pd
from transformers import pipeline

# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "results", "models_upgraded")

RAW_CSV_FALLBACK = os.path.join(DATA_DIR, "all_tickets_processed_improved_v3.csv")

MODEL_PREFIX = "ticket_classifier_model_"
VECT_PREFIX  = "ticket_tfidf_vectorizer_"

# ============================================================
# SUB-LABEL DEFINITIONS
# ============================================================

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

# ============================================================
# LOAD COMPONENTS
# ============================================================

def get_latest_by_prefix(directory, prefix):
    files = [f for f in os.listdir(directory)
             if f.startswith(prefix) and f.endswith(".joblib")]
    if not files:
        raise RuntimeError(f"No {prefix}* joblib found.")
    return os.path.join(directory, max(files))


def load_model_and_vectorizer():
    model_path = get_latest_by_prefix(MODELS_DIR, MODEL_PREFIX)
    vect_path  = get_latest_by_prefix(MODELS_DIR, VECT_PREFIX)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    print(f"[LOAD] Model      → {model_path}")
    print(f"[LOAD] Vectorizer → {vect_path}")

    return model, vectorizer


def load_label_list():
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        return sorted(df["label"].unique().tolist())

    df = pd.read_csv(RAW_CSV_FALLBACK)
    return sorted(df["label"].unique().tolist())


# ============================================================
# HUGGINGFACE SUBLABEL PIPELINE
# ============================================================

def load_zero_shot_pipeline():
    print("\n[HF] Loading zero-shot model (facebook/bart-large-mnli)...")
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    print("[HF] Loaded.")
    return clf


# ============================================================
# MAIN LABEL VIA LOGISTIC REGRESSION ONLY
# ============================================================

def predict_main_label(text, lr_model, vectorizer):
    X = vectorizer.transform([text])
    pred = lr_model.predict(X)[0]
    return pred


# ============================================================
# SUB-LABEL (HF ONLY)
# ============================================================

def predict_sub_label(text, main_label, hf_clf):
    candidates = SUB_LABELS.get(main_label)
    if not candidates:
        return None, None

    hf_output = hf_clf(text, candidate_labels=candidates, multi_label=True)
    return hf_output["labels"][0], hf_output["scores"][0]


# ============================================================
# OPTIONAL EVALUATION (LR ONLY)
# ============================================================

def load_test_texts_and_labels():
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if os.path.exists(test_path):
        df = pd.read_csv(test_path)
        return df["Description_clean"].tolist(), df["label"].tolist()

    df = pd.read_csv(RAW_CSV_FALLBACK)
    start = int(len(df) * 0.8)
    df_test = df.iloc[start:]
    return df_test["Description_clean"].tolist(), df_test["label"].tolist()


def evaluate_lr_only(lr_model, vectorizer):
    from sklearn.metrics import classification_report

    texts, labels_true = load_test_texts_and_labels()
    X = vectorizer.transform(texts)
    preds = lr_model.predict(X)

    print("\n=== LR-ONLY MODEL ACCURACY ===\n")
    print(classification_report(labels_true, preds, digits=3))


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(lr_model, vectorizer, hf_clf):
    print("\n=== Ticket Classifier (Main = LR, Sub = HF) ===\n")

    while True:
        text = input("Ticket description: ").strip()
        if text.lower() in ("exit", "quit"):
            break

        if len(text) < 3:
            print("Please type more text.\n")
            continue

        main_label = predict_main_label(text, lr_model, vectorizer)
        sub_label, sub_score = predict_sub_label(text, main_label, hf_clf)

        print(f"\n→ MAIN PREDICTION: {main_label}")
        if sub_label:
            print(f"→ SUB-CATEGORY : {sub_label}  (≈ {sub_score:.2f})")
        print("")


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    lr_model, vectorizer = load_model_and_vectorizer()
    hf_clf = load_zero_shot_pipeline()

    print("\nSelect mode:")
    print(" 1) Interactive prediction")
    print(" 2) Evaluate LR model only")
    choice = input("Choice [1/2]: ").strip()

    if choice == "2":
        evaluate_lr_only(lr_model, vectorizer)
    else:
        interactive_mode(lr_model, vectorizer, hf_clf)


if __name__ == "__main__":
    main()
