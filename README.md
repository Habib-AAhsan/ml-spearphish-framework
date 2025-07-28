# 🛡️ ML Spear-Phishing Detection Framework

An intelligent, multi-phase spear-phishing detection project designed with commercial scalability and cybersecurity integration in mind.

---

## 🚀 Project Overview

This project implements a hybrid ML system to detect spear-phishing emails using:

- TF-IDF vectorization
- Phishing-aware engineered features
- XGBoost classifier
- Flask API for inference
- Prometheus for monitoring
- Grafana for dashboard visualization
- Docker for containerization

---

## 🧠 Multi-Phase Roadmap

### ✅ Phase 1: Minimum Viable Product (MVP)
- ✔️ Built with TF-IDF + XGBoost
- ✔️ Containerized Flask API
- ✔️ Prometheus + Grafana live monitoring
- ✔️ Tested on real phishing dataset
- ✔️ Hosted locally via Docker Compose
- ✔️ Grafana dashboard showing prediction count

### 🔜 Phase 2: Deep NLP
- Integration of BERT/DistilBERT for semantic detection

### 🔄 Phase 3: Adaptive Learning
- Human-in-the-loop, live feedback API, and retraining

### 🔁 Phase 4: Proactive Threat Response
- VirusTotal / AbuseIPDB integration, automated response engine

---

## 🧰 Tools & Technologies

| Category             | Tools/Tech Stack                                                 |
|----------------------|------------------------------------------------------------------|
| ML & NLP             | Python, XGBoost, scikit-learn, TF-IDF, Pandas, NumPy             |
| API & UI             | Flask, Streamlit                                                 |
| Monitoring           | Prometheus, Grafana                                              |
| Deployment           | Docker, Docker Compose                                           |
| Dev Tools            | Git, GitHub, GitHub Actions (planned), VS Code                  |
| OS Platforms         | macOS (dev), Linux (Docker image)                               |

---

## 📦 Folder Structure

Run this to generate:
```bash
tree -I '__pycache__|venv|.git' -L 3 > structure.txt
```

📄 [`structure.txt`](structure.txt)

---

## 📈 Prometheus Metrics

The API exposes the following metrics at `http://localhost:8000/metrics`:

- `phishing_predictions_total`: Counter for prediction attempts
- (Optional future): `http_requests_total`: Track API hits

Grafana visualizes this data by querying Prometheus inside Docker:
```prometheus
phishing_predictions_total
```

---

## 🧪 Example Usage

Send a POST request to:
```
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "Dear user, your account has been compromised. Please login now."
}
```

---

## 🔒 Commercial Security Intent

This is a prototype for a self-improving spear-phishing defense system with real-time detection, feedback integration, and threat intelligence sourcing — with potential SaaS deployment in the future.

---

## 👨‍💻 Project Maintainer

A. Ahsan (HABIB) — Data Engineering & Cybersecurity Enthusiast  
[GitHub Profile](https://github.com/your-username)  
[LinkedIn](https://linkedin.com/in/your-link)

---

_Last updated: July 2025_