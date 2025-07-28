from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import sqlite3
# import for prediction logs to be dowloaded as CSV
from flask import send_file
import csv
import io
# Prometheus client for monitoring
from prometheus_client import Counter, generate_latest


# Initialize DB and table (if not exist)
conn = sqlite3.connect('logs/predictions.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_text TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()


app = Flask(__name__)
# Prometheus metrics
phishing_counter = Counter('phishing_predictions_total', 'Total phishing predictions made')
legit_counter = Counter('legit_predictions_total', 'Total legit predictions made')
all_counter = Counter('total_predictions', 'Total predictions made')



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

def log_prediction(email_text, prediction, confidence):
    cursor.execute(
        "INSERT INTO prediction_logs (email_text, prediction, confidence) VALUES (?, ?, ?)",
        (email_text, prediction, confidence)
    )
    conn.commit()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = note = None
    if request.method == "POST":
        email = request.form["emailText"]
        X = extract_features(email)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        prediction = "Phishing" if pred == 1 else "Legit"
        confidence = round(float(prob * 100), 2)

        # Prometheus metrics increment
        if pred == 1:
            phishing_counter.inc()
        else:
            legit_counter.inc()
        all_counter.inc()

        log_prediction(email, prediction, confidence)
        
        if confidence < 70:
            note = "⚠️ Low confidence — review recommended."

    return render_template("index.html", prediction=prediction, confidence=confidence, note=note)
# prediction route
@app.route("/download-csv")
def download_csv():
    cursor.execute("SELECT * FROM prediction_logs ORDER BY id DESC")
    rows = cursor.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'email_text', 'prediction', 'confidence', 'timestamp'])  # header
    writer.writerows(rows)

    output.seek(0)
    return send_file(
        io.BytesIO(output.read().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='prediction_logs.csv'
    )


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}


# if __name__ == "__main__":
#     # app.run(debug=True)
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

from prometheus_client import start_http_server

if __name__ == "__main__":
    import os

    # Start Prometheus metrics server on port 8000 (exposes /metrics)
    start_http_server(8000)

    # Start the main Flask app
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)

    
