# feature_engineering.py
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv('data/phishing_email_clean.csv')
df = df.dropna(subset=['clean_text'])

# Define phishing keywords
keywords = ["verify", "account", "urgent", "click", "login", "password", "alert", "confirm", "bank", "security"]

def extract_custom_features(df):
    features = pd.DataFrame()

    features['email_length'] = df['clean_text'].apply(len)
    features['num_words'] = df['clean_text'].apply(lambda x: len(x.split()))
    features['num_links'] = df['clean_text'].apply(lambda x: len(re.findall(r'http[s]?://', x)))
    features['num_uppercase_words'] = df['clean_text'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    features['num_exclamations'] = df['clean_text'].apply(lambda x: x.count('!'))
    features['num_digits'] = df['clean_text'].apply(lambda x: sum(c.isdigit() for c in x))
    features['contains_keywords'] = df['clean_text'].apply(
        lambda x: sum(1 for kw in keywords if kw in x.lower())
    )
    
    return features

# Custom features
custom_features = extract_custom_features(df)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Combine TF-IDF + custom
X_combined = np.hstack((X_tfidf.toarray(), custom_features.values))

# Labels
y = df['label'].values  # assuming your label column is named 'label'

# Save for next stage
np.save('data/X_combined.npy', X_combined)
np.save('data/y_labels.npy', y)

print("✅ Feature extraction complete. Shapes:")
print(f"X_combined: {X_combined.shape}")
print(f"y: {y.shape}")
