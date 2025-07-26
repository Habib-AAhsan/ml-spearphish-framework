# train_baseline_model.py
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load features and labels
X = np.load('data/X_combined.npy')
y = np.load('data/y_labels.npy')

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optionally scale non-TFIDF features (last 7 cols)
scaler = StandardScaler()
X_train[:, -7:] = scaler.fit_transform(X_train[:, -7:])
X_test[:, -7:] = scaler.transform(X_test[:, -7:])

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/logistic_regression_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\n✅ Model and scaler saved to /model/")
