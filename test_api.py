# test_api.py
import requests

email = {
    "text": "Dear user, your account has been compromised. Click here to reset your password immediately."
}

response = requests.post("http://127.0.0.1:5000/predict", json=email)

print("📬 Server Response:")
print(response.json())
