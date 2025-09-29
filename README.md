# Diabetes Classification 

## What it includes
- `main.py` — one-file pipeline: EDA → split → 3 models (LogReg, Gradient Boosting, Decision Tree) → metrics → ROC → feature importance → reduced-feature test → threshold tuning.
- `outputs/` — figures (`cm_*.png`, `roc_all_models.png`) and tables (`model_results.csv`, `*_top_*.csv`, etc.).

## How to run
```bash
python main.py --data diabetes_binary_classification_data.csv --outputs outputs
```

## project structure
```
.
├── main.py
├── diabetes_binary...csv
├── README.md
└── outputs/
```

## Notes
- Uses **class weights** to address imbalance; also shows **threshold tuning** to minimize false negatives.

