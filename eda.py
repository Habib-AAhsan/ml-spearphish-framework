# eda.py - Exploratory Data Analysis on Cleaned Phishing Email Dataset

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ 1. Load the cleaned dataset safely
clean_data_path = "data/phishing_email_clean.csv"
if not os.path.exists(clean_data_path):
    print(f"❌ File not found: {clean_data_path}")
    exit(1)

df = pd.read_csv(clean_data_path)
df['clean_text'] = df['clean_text'].fillna('')

# ✅ 2. Class Distribution
print("\n--- Class Distribution ---")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))

# ✅ 3. Text Length Stats
df['text_length'] = df['clean_text'].apply(len)
print("\n--- Text Length Stats by Class ---")
print(df.groupby('label')['text_length'].describe())

# ✅ 4. Interactive Word Frequency Plot
def plot_top_words(label, n=20):
    """Interactive bar plot of top N frequent words for given class."""
    words = ' '.join(df[df['label'] == label]['clean_text']).split()
    counter = Counter(words)
    top_words = counter.most_common(n)
    if not top_words:
        print(f"No words found for label {label}")
        return
    words, counts = zip(*top_words)
    
    fig = px.bar(
        x=words,
        y=counts,
        labels={'x': 'Word', 'y': 'Frequency'},
        title=f"Top {n} Words in {'Phishing' if label == 1 else 'Legitimate'} Emails"
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

plot_top_words(0)  # Legitimate
plot_top_words(1)  # Phishing

# ✅ 5. TF-IDF Analysis
def tfidf_top_terms(label, top_n=20):
    """Print top TF-IDF terms for the selected class."""
    texts = df[df['label'] == label]['clean_text']
    if texts.empty:
        print(f"No text found for label {label}")
        return
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    top_indices = mean_tfidf.argsort()[::-1][:top_n]
    top_terms = [(terms[i], mean_tfidf[i]) for i in top_indices]

    print(f"\nTop TF-IDF Terms for {'Phishing' if label == 1 else 'Legitimate'} Emails:")
    for word, score in top_terms:
        print(f"{word}: {score:.4f}")

tfidf_top_terms(0)  # Legitimate
tfidf_top_terms(1)  # Phishing

# ✅ 6. WordClouds
def generate_wordcloud(label):
    text = ' '.join(df[df['label'] == label]['clean_text'])
    if not text.strip():
        print(f"No content to generate wordcloud for label {label}")
        return
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{'Phishing' if label == 1 else 'Legitimate'} Email WordCloud")
    plt.tight_layout()
    plt.show()

generate_wordcloud(0)  # Legitimate
generate_wordcloud(1)  # Phishing



# ✅ What This Script Does

# ✔️ Loads the Cleaned Dataset
# df = pd.read_csv("data/phishing_email_clean.csv")
# df['clean_text'] = df['clean_text'].fillna('')
# Uses clean_text column — ✅ matches your preprocessing output.
# Assumes a label column exists — ✅ OK if you included it in the saved file.
# ✔️ Performs Class Distribution Analysis
# print(df['label'].value_counts())
# print(df['label'].value_counts(normalize=True))
# Useful for spotting imbalance.
# No issues here.
# ✔️ Computes Text Length Statistics
# df['text_length'] = df['clean_text'].apply(len)
# print(df.groupby('label')['text_length'].describe())
# Helps compare verbosity of phishing vs legit emails.
# ✔️ Generates Top Word Bar Charts (Using plotly)
# plot_top_words(0)
# plot_top_words(1)
# ✅ Very helpful visualization.
# Good use of Counter, plotly.express, and label mapping.
# ✔️ TF-IDF Keyword Extraction
# tfidf_top_terms(0)
# tfidf_top_terms(1)
# Captures top discriminative terms per class.
# ✅ Efficient TfidfVectorizer use.
# ✔️ WordClouds for Each Class
# generate_wordcloud(0)
# generate_wordcloud(1)
# Leverages visual insight.
# Also handles empty cases gracefully.
# 🛠️ Suggestions & Best Practices

# Area	Recommendation
# 🔒 File Check	Consider wrapping all df['label'] usage in if 'label' in df.columns: to avoid crashing on missing label
# 📂 Flexibility	Use argparse or sys.argv for dynamic file path selection in CLI
# 📈 Plots	Optionally export plots to PNG using fig.write_image() or plt.savefig()
# 🧼 Pre-check	Add df.dropna(subset=["clean_text"], inplace=True) before analysis
# 📊 Bonus	Add a histogram: df['text_length'].hist(by=df['label']) to compare length distribution


# AFTER SUCCESSFUL EXECUTION:

# What Your EDA Output Shows:
# 1. Class Distribution

# Phishing (1): 42,891 emails (≈52%)
# Legitimate (0): 39,595 emails (≈48%)
# Balanced enough for ML training!

# 2. Text Length Stats

# Legitimate emails are longer on average.
# Phishing emails vary widely — some very short, others ridiculously long (could be obfuscation).
# 3. Top TF-IDF Terms

# Legitimate: enron, ect, thanks, message, file, hou
# Phishing: replica, click, alert, cnncom, money, account
# This confirms phishing indicators like “replica,” “money,” “click,” are strong signals.

# ✅ What You’ve Done So Far:
# Step	Task	Status
# 1	Preprocessing fixed (NLTK, stopwords, clean_text)	✅ Completed
# 2	Ensured punkt and other NLTK resources are working offline	✅ Completed
# 3	Cleaned dataset saved as data/phishing_email_clean.csv	✅ Done
# 4	EDA script eda.py created and tested	✅ Done
# 5	Required packages installed (matplotlib, plotly, wordcloud, etc.)	✅ Done
# 🪜 What You Can Do Next:
# Task	Description
# 📈 Feature Engineering	e.g., email length, TF-IDF vectors, sender domain, presence of links
# 🧪 Train/Test Split	Prepare datasets for ML training
# 🤖 Model Building	Logistic Regression, Random Forest, SVM, etc.
# 📊 Evaluation	Accuracy, Precision, Recall, F1-score
# 🚀 Deployment Prep	Save pipeline, test inference, consider Flask API, etc.