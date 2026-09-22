# Data

This folder is intentionally **not** committed to git (see `.gitignore`).
The raw file and every derived split together total ~1.9GB, which is far
past what a git repo should carry and over GitHub's 100MB per-file limit
for several of the split files. Everything under `data/` is regenerated
from one downloaded CSV by the scripts in `src/`.

## 1. Download the raw dataset

Source: [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
on Kaggle (`kartik2112/fraud-detection`). It's a synthetic dataset generated
with the `Sparkov` transaction simulator — the PII-shaped columns (`first`,
`last`, `street`, `cc_num`, `dob`, etc.) are fabricated, not real people.

```bash
# One-time: install the Kaggle CLI and put your API token at ~/.kaggle/kaggle.json
# (Kaggle account -> Settings -> API -> "Create New Token")
pip install kaggle

# Download and unzip into data/raw/
mkdir -p data/raw
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

This produces `data/raw/fraudTrain.csv` (the file `src/config.py` expects
at `RAW_DATA_FILE`). If Kaggle names the extracted file differently, rename
it to `fraudTrain.csv`.

You can also download it manually from the Kaggle page above (click
"Download") if you'd rather not set up the CLI/API token.

## 2. Regenerate everything else

Run these from the repo root, in order, with your virtualenv active:

```bash
python -m src.load_data                    # -> data/samples/fraud_dev_sample.csv
python -m src.enrich_data                   # -> data/processed/fraud_enriched_sample.csv
python -m src.build_features                # -> data/processed/splits/{train,validation,test}.csv + models/preprocessor.joblib
python -m src.build_features_random_split   # -> *_full_random.csv variant
```

See the main `README.md` for the full pipeline (training, tuning, ablation,
and figure generation) that runs after this step.
