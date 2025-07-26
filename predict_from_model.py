# predict_from_model.py
import re
import numpy as np
import joblib
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Step 1: Load saved model and scaler ---
MODEL_PATH = 'model/xgboost_model.pkl'
SCALER_PATH = 'model/xgb_scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Step 2: Load and rebuild TF-IDF vectorizer with training vocab ---
# You must use the same vocabulary as used in training
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('data/phishing_email_clean.csv')

# 🔍 Check and drop NaN rows safely
if df['clean_text'].isnull().any():
    print("⚠️ Warning: Missing values found in 'clean_text'. Dropping them.")
    df = df.dropna(subset=['clean_text'])

tfidf = TfidfVectorizer(max_features=1000)
tfidf.fit(df['clean_text'])

# --- Step 3: Custom phishing features ---
phishing_keywords = ["verify", "account", "urgent", "click", "login", "password", "alert", "confirm", "bank", "security"]

def extract_custom_features(text):
    features = []
    features.append(len(text))  # email_length
    features.append(len(text.split()))  # num_words
    features.append(len(re.findall(r'http[s]?://', text)))  # num_links
    features.append(sum(1 for w in text.split() if w.isupper()))  # num_uppercase
    features.append(text.count('!'))  # num_exclamations
    features.append(sum(c.isdigit() for c in text))  # num_digits
    features.append(sum(1 for kw in phishing_keywords if kw in text.lower()))  # contains_keywords
    return features

# --- Step 4: Preprocess text (mirror your training cleaner) ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Step 5: Inference function ---
def predict(email_text):
    clean_text = preprocess(email_text)
    tfidf_vec = tfidf.transform([clean_text]).toarray()
    custom_vec = np.array(extract_custom_features(clean_text)).reshape(1, -1)
    custom_vec = scaler.transform(custom_vec)
    X_final = np.hstack((tfidf_vec, custom_vec))
    
    pred = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0][1] if hasattr(model, "predict_proba") else None

    print("\n📩 Prediction:", "Phishing" if pred == 1 else "Legit")
    if prob is not None:
        print("🔢 Confidence (Phishing): {:.2f}%".format(prob * 100))

# --- Step 6: CLI for testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input email text")
    args = parser.parse_args()
    predict(args.text)
