import os
import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("✅ Data Loaded!")
        print("📊 Shape:", df.shape)
        print("🔍 Columns:", df.columns.tolist())
        print("\nSample rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None

if __name__ == "__main__":
    load_data("data/phishing_email.csv")


# | Feature                     | Status | Notes                                      |
# | --------------------------- | ------ | ------------------------------------------ |
# | **Function-based loading**  | ✅ Good | `load_data()` is reusable in other scripts |
# | **Error handling**          | ✅ Good | Graceful `FileNotFoundError` fallback      |
# | **Main guard** (`__main__`) | ✅ Good | Keeps behavior controlled when importing   |
# | **Prints useful info**      | ✅ Good | Shape, columns, and sample rows            |
# | **Path correct**            | ✅ OK   | Assumes file is in `data/` folder          |

