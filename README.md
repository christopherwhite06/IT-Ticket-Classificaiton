# IT-Ticket-Classificaiton
Built an end-to-end ML pipeline to classify IT service tickets (Access, Hardware, HR Support, Storage, etc.) using TF-IDF and multinomial Logistic Regression. Automatically labels incoming tickets to improve routing accuracy and reduce response times.

- Link to dataset used: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset?resource=download

- Resulting accuracy: 86% based on main labels. There are sub-labels used for expressing the ticket issue in more detail.

## Pipeline Overview
1. **Data Processing (`process_data.py`)**  
   - Loads raw ticket CSV  
   - Cleans and normalises text  
   - Creates `train.csv` and `test.csv` (stratified split)

2. **Feature Extraction (`extract_features.py`)**  
   - Applies TF-IDF vectorisation (1–2 grams, up to 20k–30k features)  
   - Saves sparse matrices and labels into `data/features/`

3. **Model Training (`train_model.py`)**  
   - Builds TF-IDF features  
   - Trains a multinomial Logistic Regression classifier  
   - Runs Grid Search  
   - Outputs metrics and saves model + vectorizer

4. **Prediction (`predict_ticket.py`)**  
   - Loads latest trained model  
   - Predicts main category  
   - Optional sub-label prediction using BART-MNLI zero-shot model  
   - Supports interactive mode and evaluation mode in the terminal

5. **Frontend GUI (`frontend.py`)**
    - Uses stream lit to run some GUI for my ticket classifier
    Outputs graphs of the probability it is a certain label.
    
   ## How to Run

### 1. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
# OR
.\.venv\Scripts\activate     # Windows

### 2. Install Required Packages

- pip install -r requirements.txt

### 3. Run the ticket classifer on the terminal or Browser


#### For browser GUI
- streamlit run src/frontend.py 

#### For within the terminal
- python src/predict_ticket.py

## Optional: Full Pipeline Execution

The repository already contains a processed dataset (`train.csv` and `test.csv`), a trained Logistic Regression model, and a saved TF-IDF vectorizer. This means you do not need to run the full pipeline unless you want to regenerate the data or retrain the model.

If you choose to manually run the entire pipeline, execute the scripts in the following order:

---

### 1. Process the raw CSV file  
This step cleans and normalises ticket text, verifies required columns, and performs an 80/20 stratified train-test split.

Command to run: `python src/process_data.py`  
Outputs:
- `data/processed/train.csv`
- `data/processed/test.csv`

---

### 2. Extract TF-IDF features  
Fits a TF-IDF vectorizer on the training text and transforms both train and test sets.

Command to run: `python src/extract_features.py`  
Outputs:
- `tfidf_vectorizer.joblib`
- `X_train_tfidf.npz`
- `X_test_tfidf.npz`
- `y_train.npy`
- `y_test.npy`

---

### 3. Train the Logistic Regression classifier  
Runs grid search, evaluates performance, and saves the best-performing model and vectorizer.

Command to run: `python src/train_model.py`  
Outputs:
- `ticket_classifier_model_<timestamp>.joblib`
- `ticket_tfidf_vectorizer_<timestamp>.joblib`

---

### 4. Predict or evaluate  
Allows interactive ticket classification or LR-only evaluation using the saved model and vectorizer.

Command to run: `python src/predict_ticket.py`  
Modes available:
- Interactive prediction  
- Evaluation on the test set  

### 5. Classify tickets with GUI and show results with graphs
Allows the program to run in the browser.

Command to run: `streamlit run src/frontend.py`

---

Running these steps is optional; the system is ready to use immediately using only `streamlit run src/frontend.py` or `predict_ticket.py`.