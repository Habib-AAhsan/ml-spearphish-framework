from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/xgboost_model.pkl")
scaler = joblib.load("model/xgb_scaler.pkl")

# Load and fit TF-IDF
import os
# if os.getenv("RENDER") != "true":
#     df = pd.read_csv("data/phishing_email_clean.csv")

# # df = pd.read_csv("data/phishing_email_clean.csv")
#     df = df.dropna(subset=["clean_text"])
#     tfidf = TfidfVectorizer(max_features=1000)
#     tfidf.fit(df["clean_text"])
tfidf = joblib.load("model/xgb_tfidf.pkl")

# Define features
phishing_keywords = ["verify", "account", "urgent", "click", "login", "password", "alert", "confirm", "bank", "security"]

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_features(text):
    clean = preprocess(text)
    tfidf_vec = tfidf.transform([clean]).toarray()
    custom = [
        len(clean),
        len(clean.split()),
        len(re.findall(r"http[s]?://", clean)),
        sum(1 for w in clean.split() if w.isupper()),
        clean.count("!"),
        sum(c.isdigit() for c in clean),
        sum(1 for kw in phishing_keywords if kw in clean)
    ]
    custom_scaled = scaler.transform([custom])
    return np.hstack((tfidf_vec, custom_scaled))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = note = None
    if request.method == "POST":
        email = request.form["emailText"]
        X = extract_features(email)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        prediction = "Phishing" if pred == 1 else "Legit"
        confidence = round(prob * 100, 2)
        if confidence < 70:
            note = "⚠️ Low confidence — review recommended."

    return render_template("index.html", prediction=prediction, confidence=confidence, note=note)

# if __name__ == "__main__":
#     # app.run(debug=True)
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
