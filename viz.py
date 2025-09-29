# viz.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_binary_classification_data.csv")
df.columns = [c.strip() for c in df.columns]
rates = df.groupby("GenHlth")["Diabetes_binary"].mean().sort_index()

plt.figure(figsize=(7,4))
rates.plot(kind="bar")
plt.ylabel("Share with pre/diabetes")
plt.xlabel("GenHlth (1=Excellent … 5=Poor)")
plt.title("Diabetes Prevalence by Self-Reported General Health")
plt.tight_layout()
plt.savefig("outputs/viz_prevalence_by_genhlth.png", dpi=150)

"mkdir -p outputs
"py viz.py"