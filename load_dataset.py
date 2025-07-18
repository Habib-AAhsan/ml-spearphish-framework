import pandas as pd

df = pd.read_csv("phishing_email.csv")  # Update filename if different

print("✅ Data Loaded!")
print("📊 Shape:", df.shape)
print("🔍 Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head())
