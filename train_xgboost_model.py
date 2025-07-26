# train_xgboost_model.py
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
X = np.load('data/X_combined.npy')
y = np.load('data/y_labels.npy')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale last 7 features (custom ones)
scaler = StandardScaler()
X_train[:, -7:] = scaler.fit_transform(X_train[:, -7:])
X_test[:, -7:] = scaler.transform(X_test[:, -7:])

# Train XGBoost classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    # use_label_encoder=False,
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model + scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/xgboost_model.pkl')
joblib.dump(scaler, 'model/xgb_scaler.pkl')

print("\n✅ XGBoost model and scaler saved to /model/")
