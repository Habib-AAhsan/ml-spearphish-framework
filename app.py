# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Load Model and Scaler ---
model = joblib.load('model/xgboost_model.pkl')
scaler = joblib.load('model/xgb_scaler.pkl')

# --- Load training data for TF-IDF vocabulary ---
df = pd.read_csv('data/phishing_email_clean.csv')
df = df.dropna(subset=['clean_text'])
tfidf = TfidfVectorizer(max_features=1000)
tfidf.fit(df['clean_text'])

# --- Define phishing-aware features ---
phishing_keywords = ["verify", "account", "urgent", "click", "login", "password", "alert", "confirm", "bank", "security"]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in JSON.'}), 400

    email_text = data['text']
    clean_text = preprocess(email_text)

    tfidf_vec = tfidf.transform([clean_text]).toarray()
    custom_vec = np.array(extract_custom_features(clean_text)).reshape(1, -1)
    custom_vec = scaler.transform(custom_vec)

    X = np.hstack((tfidf_vec, custom_vec))

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

    label = "Phishing" if pred == 1 else "Legit"
    confidence = round(prob * 100, 2) if prob is not None else None

    result = {
        "prediction": label,
        "confidence": f"{confidence}%" if confidence is not None else "N/A"
    }

    # Optional soft logic
    if confidence is not None and confidence < 70:
        result["note"] = "Low confidence — review recommended"

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
