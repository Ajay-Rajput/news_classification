# News Classification Project

Simple scaffold for a news classification pipeline.

Structure:
- `data/raw/train.csv` — raw training CSV
- `data/processed/cleaned.csv` — cleaned data
- `src/` — data preprocessing, feature engineering, train, evaluate
- `models/news_classifier.pkl` — trained model (placeholder)
- `results/metrics.txt` — evaluation output

Quick start:

1. Create a virtualenv and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run pipeline from project root:

```powershell
python main.py
```
