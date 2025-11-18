import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import datetime

# paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results", "models_upgraded")
os.makedirs(RESULTS_DIR, exist_ok=True)

# load data
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

X_train = train_df["Description_clean"]
y_train = train_df["label"]
X_test = test_df["Description_clean"]
y_test = test_df["label"]

print("Loaded data:")
print(f"  Train rows: {len(train_df)}")
print(f"  Test rows : {len(test_df)}")

# tfidf
print("\n=== Building TF-IDF features (this can take a bit) ===")
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english",
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF complete.")
print("Train shape:", X_train_vec.shape)
print("Test shape:", X_test_vec.shape)

# logistic regression grid search
model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="saga",
    multi_class="multinomial",
    n_jobs=-1,
)

param_grid = {"C": [3.0]}

print("\n=== Running Grid Search (this is the slowest step) ===")
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_weighted",
    n_jobs=-1,
    cv=3,
    verbose=3
)

grid.fit(X_train_vec, y_train)
best_model = grid.best_estimator_

print("\n=== BEST MODEL FOUND ===")
print(best_model)

# eval
print("\n=== Evaluating on test set ===\n")
y_pred = best_model.predict(X_test_vec)
print(classification_report(y_test, y_pred, digits=3))

# save
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(RESULTS_DIR, f"ticket_classifier_model_{timestamp}.joblib")
vect_path = os.path.join(RESULTS_DIR, f"ticket_tfidf_vectorizer_{timestamp}.joblib")

joblib.dump(best_model, model_path)
joblib.dump(vectorizer, vect_path)

print("\nSaved model ->", model_path)
print("Saved vectorizer ->", vect_path)