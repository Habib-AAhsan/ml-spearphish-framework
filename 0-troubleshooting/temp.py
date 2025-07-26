import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/phishing_email.csv")

# Show unique labels to understand which one is phishing
print("Unique labels in the dataset:", df['label'].unique())
print("\nLabel distribution:")
print(df['label'].value_counts())

# Add a new column for text length
df['text_length'] = df['text_combined'].apply(len)

# Plot the distribution
sns.histplot(data=df, x='text_length', hue='label', bins=50)
plt.title('Text Length Distribution by Label')
plt.xlabel('Text Length (Number of Characters)')
plt.ylabel('Email Count')
plt.legend(title='Label (phishing vs non)')
plt.tight_layout()
plt.show()
