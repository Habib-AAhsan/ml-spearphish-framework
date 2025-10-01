#!/usr/bin/env python3
# scripts/load_demo.py
import json, sys

def simple_score(body: str):
    body = body.lower()
    if "invoice" in body or "click the link" in body:
        return 0.92
    if "patient" in body:
        return 0.3
    return 0.1

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "data/toy.json"
    with open(fname) as f:
        data = json.load(f)
    out = [{"id": d["id"], "score": simple_score(d.get("body",""))} for d in data]
    print(json.dumps(out, indent=2))
