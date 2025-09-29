# Diabetes Classification — Starter Pack

This repo-ready starter shows you the **type of code** to write and gives you a runnable baseline.

## What it includes
- `main.py` — one-file pipeline: EDA → split → 3 models (LogReg, Gradient Boosting, Decision Tree) → metrics → ROC → feature importance → reduced-feature test → threshold tuning.
- `outputs/` — figures (`cm_*.png`, `roc_all_models.png`) and tables (`model_results.csv`, `*_top_*.csv`, etc.).

## How to run
```bash
python main.py --data /path/to/diabetes_binary_classification_data.csv --outputs outputs
```

## Extend for your deliverables
- Add a `notebooks/analysis.ipynb` if your course prefers a notebook.
- For the report, answer the 8 questions using numbers/plots saved in `outputs/`.
- For collaboration, add a short `COLLAB.md` (≤1000 words).

## Suggested project structure
```
.
├── main.py
├── README.md
└── outputs/
```

## Notes
- Uses **class weights** to address imbalance; also shows **threshold tuning** to minimize false negatives.
- No deep learning, only off-the-shelf scikit-learn models.
