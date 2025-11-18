import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
FEATURES_DIR = os.path.join(ROOT, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_PROCESSED, "train.csv")
TEST_CSV = os.path.join(DATA_PROCESSED, "test.csv")

def main():
    # load data
    print("\n=== LOADING CLEAN TRAIN/TEST ===")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")

    # extract text/labels
    X_train_text = train_df["Description_clean"].astype(str).tolist()
    X_test_text = test_df["Description_clean"].astype(str).tolist()
    y_train = train_df["label"].astype(str).values
    y_test = test_df["label"].astype(str).values

    # tfidf
    print("\n=== FITTING TF-IDF VECTORIZER ===")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # save
    print("\n=== SAVING FEATURES ===")
    joblib.dump(vectorizer, os.path.join(FEATURES_DIR, "tfidf_vectorizer.joblib"))

    from scipy.sparse import save_npz
    save_npz(os.path.join(FEATURES_DIR, "X_train_tfidf.npz"), X_train_tfidf)
    save_npz(os.path.join(FEATURES_DIR, "X_test_tfidf.npz"), X_test_tfidf)

    np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)

    print("Saved:")
    print("→ tfidf_vectorizer.joblib")
    print("→ X_train_tfidf.npz")
    print("→ X_test_tfidf.npz")
    print("→ y_train.npy")
    print("→ y_test.npy")

    print("\nFeature engineering complete.\n")

if __name__ == "__main__":
    main()