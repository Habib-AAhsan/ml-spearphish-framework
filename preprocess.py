import os, re, nltk, pandas as pd

nltk.data.path.append('./nltk_data')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK setup (run once, then comment out)
# nltk.download("stopwords")
# nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def preprocess_text(text, verbose=False):
    if not isinstance(text, str) or not text.strip():
        return ""

    if verbose: print(f"\n🔤 Original: {text}")

    try:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w not in stop_words and len(w) > 2]

        final_text = ' '.join(filtered).strip()

        if verbose:
            print(f"✅ Cleaned text: '{final_text}'")

        return final_text
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""

def preprocess_dataframe(df, text_column="text_combined", verbose=False):
    df = df.copy()
    df["clean_text"] = df[text_column].astype(str).apply(lambda x: preprocess_text(x, verbose))

    return df

if __name__ == "__main__":
    input_path = "data/phishing_email.csv"
    output_path = "data/phishing_email_clean.csv"

    df = pd.read_csv(input_path)
    sample_df = df.head(10)
    preprocess_dataframe(sample_df, verbose=True)

    proceed = input("\n➡️ Process full dataset? (y/n): ")
    if proceed.lower() == 'y':
        df = preprocess_dataframe(df)
        df.to_csv(output_path, index=False)
        print("✅ Done preprocessing!")
    else:
        print("🛑 Cancelled.")





# | Feature                     | Benefit                                             |
# | --------------------------- | --------------------------------------------------- |
# | `preprocess_text()`         | Reusable text-cleaning logic                        |
# | `preprocess_dataframe()`    | Keeps logic separate and readable                   |
# | Local `nltk_data` directory | Ensures downloads are contained within your project |
# | Main section (`__main__`)   | Clean CLI execution with fallback for missing files |
# | Consistent paths            | Reads from and writes to your `data/` directory     |


# | Step                          | Description                                                        |
# | ----------------------------- | ------------------------------------------------------------------ |
# | **1. Load Data**              | Loads `phishing_email.csv` from the `data/` directory              |
# | **2. NLTK Setup**             | Downloads `stopwords` and `punkt` into a local `nltk_data/` folder |
# | **3. Preprocessing Function** | Cleans text: lowercase, removes URLs, punctuation, stopwords       |
# | **4. Apply Preprocessing**    | Adds a new `clean_text` column                                     |
# | **5. Save Cleaned File**      | Writes cleaned data to `phishing_email_clean.csv`                  |
# | **6. Preview**                | Prints a few rows to verify changes                                |




# Reasons preprocess.py is slow:
# Row-wise processing:
# df['clean_text'] = df['text_combined'].astype(str).apply(preprocess_text)
# This applies preprocess_text() to each row individually — which is inherently slow in Python, especially without vectorization.
# NLTK Tokenization:
# tokens = word_tokenize(text)
# NLTK’s word_tokenize is accurate but slow, especially across tens of thousands of rows.
# Regex operations:
# Removing URLs
# Removing non-letter characters
# These regex replacements are fast in small scale but slow when repeated over a large dataset.
# Stopword filtering:
# Iterates through tokens to remove common English words.
# No caching or parallelization:
# All steps run serially in a single core — not parallelized.
# 🚀 How to speed it up (optional for later):
# Use spaCy or HuggingFace’s transformers tokenizer — much faster with better control.
# Parallel processing with swifter, joblib, or multiprocessing.
# Cache preprocessed data so you don't run preprocess.py again unless needed.
