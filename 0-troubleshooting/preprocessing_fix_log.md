# 🧠 Preprocessing Fix Log & Troubleshooting Guide (NLTK / `preprocess.py`)

This notebook documents the problems and solutions applied to make the `preprocess.py` script work reliably for text preprocessing in the `ml-spearphish-framework`.

---

## ✅ Problem Summary

The `preprocess.py` script was filtering **everything**, even valid words. The root cause was a broken `word_tokenize()` call from NLTK — due to a bug in recent NLTK versions.

---

## ✅ What We Did to Fix It

### 1. Identified the Bug
- NLTK threw this error:
  ```
  LookupError: Resource punkt_tab not found.
  ```
- Cause: `punkt_tab` is **not a real resource** — this is a bug introduced in NLTK ≥ 3.8.1.

---

### 2. Verified Tokenizer Error
```bash
python -c "from nltk.tokenize import word_tokenize; print(word_tokenize('Hello World!'))"
```
- Result: `LookupError` → `punkt_tab` → tokenizer broken

---

### 3. Downgraded NLTK to Fix It
```bash
pip uninstall nltk -y
pip install nltk==3.8.0
```
✅ Fixes `punkt_tab` error permanently.

---

### 4. Downloaded `punkt.zip` Manually (SSL Fix)
- macOS SSL blocked downloads.
- So we downloaded from:
  [punkt.zip (GitHub mirror)](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip)
- Extracted into:
  ```
  ./nltk_data/tokenizers/punkt/
  ```

✅ `english.pickle` now available.

---

### 5. Updated Preprocessing Code to Use Local NLTK Path
```python
import nltk
nltk.data.path.append('./nltk_data')
```

✅ Tokenizer loads from the correct path.

---

### 6. Ran `preprocess.py`
- Prompted:
  ```
  ➡️ Process full dataset? (y/n):
  ```
  - `n` = Test only first 10 rows
  - `y` = Process full dataset and save

✅ Success with both modes.

---

### 7. Fixed Pandas Warning (Optional)
To silence:
```text
SettingWithCopyWarning
```

Add:
```python
df = df.copy()
```
Before assigning:
```python
df["clean_text"] = ...
```

---

## 🧭 What to Do If This Happens Again

| Step | Action | Purpose |
|------|--------|---------|
| 1️⃣ | `python -c "from nltk.tokenize import word_tokenize; print(word_tokenize('Hello World!'))"` | Check if `punkt` works |
| 2️⃣ | If error → `pip install nltk==3.8.0` | Downgrade to stable |
| 3️⃣ | If SSL issue → manually download `punkt.zip` | Use offline |
| 4️⃣ | Add `nltk.data.path.append('./nltk_data')` | Point to local data |
| 5️⃣ | Test with `n`, run full with `y` | Confirm cleaning works |
| 6️⃣ | Add `df = df.copy()` if warning shows | Avoid Pandas side effects |

---

## 🧾 Add This to README.md or Project Log

```text
# nltk fix log
- NLTK downgraded to 3.8.0
- punkt manually installed in ./nltk_data/tokenizers/punkt/
- nltk.data.path.append('./nltk_data') added to scripts
```

---

✅ You are now ready for clean, stable, repeatable preprocessing every time.
