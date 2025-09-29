# eda.py
import pandas as pd

df = pd.read_csv("diabetes_binary_classification_data.csv")
df.columns = [c.strip() for c in df.columns]
y = 'Diabetes_binary'
print("Shape:", df.shape)
print("\nClass balance:\n", df[y].value_counts().sort_index())
print("\nMissing values (top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))
