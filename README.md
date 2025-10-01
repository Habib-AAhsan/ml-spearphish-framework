# ml-spearphish-framework
Production-minded spear-phishing detection prototype for healthcare — FastAPI demo, toy dataset, SHAP-ready explainability, and CI. ▶️ Try the demo locally: `make demo`

_Last updated: August 2025_

---

## Quickstart (demo)

Quick 5-minute demo to run locally:

```bash
# clone & start (one-time)
git clone https://github.com/Habib-AAhsan/ml-spearphish-framework.git
cd ml-spearphish-framework

# start demo (creates virtualenv, installs deps, generates toy data and runs server)
make demo

# in another terminal, test the endpoint
curl -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @data/toy.json | jq

