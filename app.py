# app.py -- minimal demo server
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="SpearPhish Demo")

class Msg(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str

@app.post("/predict")
def predict(msgs: List[Msg]):
    def score(body):
        b = body.lower()
        if "invoice" in b or "click the link" in b:
            return 0.92
        if "patient" in b:
            return 0.3
        return 0.05
    return [{"id": m.id, "score": score(m.body)} for m in msgs]

# Keep the existing root health endpoint
@app.get("/")
def health():
    return {"status": "ok"}

# Explicit readiness/liveness endpoint for Render / load-balancers
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

