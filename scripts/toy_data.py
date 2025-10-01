#!/usr/bin/env python3
# scripts/toy_data.py
import json
import argparse
import random
from datetime import datetime

def gen_sample(i):
    return {
        "id": f"msg-{i}",
        "subject": random.choice([
            "Urgent: Update Your Credentials", "Invoice Attached", "Patient Lab Results"
        ]),
        "body": random.choice([
            "Please click the link to update your bank info",
            "Your invoice is attached. Open to pay.",
            "Important: view patient lab results here"
        ]),
        "sender": random.choice(["admin@hospital.example","info@clinic.example","external@unknown.example"]),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default="data/toy.json")
    args = parser.parse_args()
    samples = [gen_sample(i) for i in range(args.n)]
    with open(args.out, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"wrote {len(samples)} samples -> {args.out}")
