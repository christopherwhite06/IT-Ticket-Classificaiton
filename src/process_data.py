import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# ===========================================
# PATHS
# ===========================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")

os.makedirs(DATA_PROCESSED, exist_ok=True)

INPUT_FILE = os.path.join(DATA_RAW, "all_tickets_processed_improved_v3.csv")


# ===========================================
# TEXT CLEANING
# ===========================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================================
# MAIN
# ===========================================

def main():

    print("=== LOADING RAW CSV ===")
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows")
    print("Columns in dataset:", df.columns.tolist())

    # ===========================================
    # VERIFY REQUIRED COLUMNS
    # ===========================================

    if "Topic_group" not in df.columns:
        raise RuntimeError("Expected column 'Topic_group' not found in CSV")

    if "Document" not in df.columns:
        raise RuntimeError("Expected text column 'Document' not found in CSV")

    # ===========================================
    # CLEAN TEXT
    # ===========================================

    print("=== Cleaning text ===")

    df["Description_clean"] = df["Document"].apply(clean_text)
    df["label"] = df["Topic_group"]

    # Remove empty rows
    df = df[df["Description_clean"].str.len() > 0]

    print("Cleaned text. Remaining rows:", len(df))

    # ===========================================
    # TRAIN TEST SPLIT (STRATIFIED)
    # ===========================================

    print("=== Splitting train and test ===")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")

    # ===========================================
    # SAVE OUTPUTS
    # ===========================================

    train_path = os.path.join(DATA_PROCESSED, "train.csv")
    test_path = os.path.join(DATA_PROCESSED, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n=== SAVED ===")
    print(f"Saved train dataset to: {train_path}")
    print(f"Saved test dataset to:  {test_path}")
    print("First program complete.\n")


if __name__ == "__main__":
    main()
