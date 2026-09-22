# Gotcha — Credit Card Fraud Detection

An end-to-end fraud detection pipeline: feature engineering on raw
transaction data, a baseline Logistic Regression model, a tuned XGBoost
model, a business-cost-aware decision threshold, and a small Flask API
that scores transactions in real time with a three-band decision policy
(approve / step-up verification / block).

## Results

Test-set performance (full feature set, XGBoost):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.598 | 0.878 | 0.712 | 0.998 | 0.900 |
| XGBoost (tuned) | 0.922 | 0.922 | 0.922 | 0.999 | 0.969 |

At the selected operating threshold (0.70, chosen to minimize expected
business cost subject to a recall floor of 0.80 and a max review rate of
0.75%): precision 0.794, recall 0.914, review rate 0.75% on the validation
set. Full metrics are in `reports/metrics/`; charts are in
`reports/figures/`.

> **Note on the numbers:** the dataset (see below) is synthetic and
> known to be relatively easy to separate — treat the near-perfect
> ROC-AUC as a property of the dataset, not a claim that this approach
> generalizes to real production fraud data.

## Project structure

```
src/
  config.py                    # centralized paths and constants
  load_data.py                 # raw CSV -> reproducible dev sample
  enrich_data.py                # feature engineering (velocity, distance, time-window aggregates)
  build_features.py            # train/val/test splits + fitted preprocessor (per variant)
  build_features_random_split.py  # same, using a plain random split instead of time-based
  train_baseline.py            # Logistic Regression baseline
  train_xgboost.py             # XGBoost model
  tune_xgboost_params.py       # hyperparameter search
  tune_threshold.py            # cost-based decision threshold selection
  run_xgboost_ablation.py      # runs the full/no_synthetic/no_ip_risk feature-variant ablation
  generate_figures.py          # renders reports/figures/ from saved metrics
  score_transaction.py         # shared decision policy (approve/step-up/block bands)
  api.py                       # Flask inference API
tests/
  test_features.py             # unit tests for feature math and decision policy
models/                        # trained model artifacts + preprocessors (committed, ~5MB)
reports/                       # metrics JSON + figures (committed, ~340KB)
data/                          # raw/processed data (gitignored — see data/README.md)
```

## Setup

Requires Python 3.14 (or a compatible 3.x).

```bash
git clone <this-repo-url>
cd Gotcha
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API (uses the pre-trained models already in `models/`)

No dataset download needed for this — the committed model artifacts are
enough to serve predictions.

```bash
cp .env.example .env
# edit .env and set a real FRAUD_API_KEY, then:
export $(cat .env | xargs)
python -m src.api
```

The API starts on `http://localhost:5001`.

```bash
curl http://localhost:5001/health

curl -X POST http://localhost:5001/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $FRAUD_API_KEY" \
  -d '{
    "amt": 125.50, "zip": 90210, "lat": 34.0522, "long": -118.2437,
    "city_pop": 500000, "merch_lat": 34.05, "merch_long": -118.25,
    "txn_hour": 14, "txn_dayofweek": 2, "is_weekend": 0, "is_night_transaction": 0,
    "customer_age": 35, "merchant_customer_distance_km": 2.3, "amount_log": 4.83,
    "amount_vs_customer_mean": 1.1, "amount_zscore_customer": 0.4,
    "time_since_last_txn_sec": 3600, "distance_from_last_txn_km": 1.2,
    "travel_speed_kmh": 5.0, "transactions_last_10min": 1,
    "synthetic_device_changed": 0, "synthetic_failed_logins_24h": 0,
    "synthetic_ip_risk_score": 0.1, "synthetic_account_age_days": 400,
    "synthetic_email_age_days": 500, "synthetic_billing_shipping_mismatch": 0,
    "category": "grocery_pos", "gender": "F", "state": "CA", "job": "Engineer"
  }'
```

Response: `fraud_probability`, `decision` (`approve` / `step_up_verification`
/ `block`), `risk_band`, and a human-readable `action`.

`MODEL_VARIANT` env var (default `full`) selects which trained variant to
serve: `full`, `full_random`, `no_synthetic`, or `no_ip_risk`.

## Run the tests

```bash
pytest
```

Tests are self-contained (synthetic inputs) and don't require the dataset.

## Reproducing the full pipeline (requires the dataset)

See `data/README.md` for how to download `fraudTrain.csv`, then:

```bash
python -m src.load_data                    # build dev sample
python -m src.enrich_data                   # engineer features
python -m src.build_features --variant full          # main train/val/test split + preprocessor
python -m src.build_features_random_split             # random-split variant
python -m src.train_baseline --variant full
python -m src.train_xgboost --variant full
python -m src.tune_xgboost_params
python -m src.tune_threshold --variant full
python -m src.run_xgboost_ablation          # full / no_synthetic / no_ip_risk comparison
python -m src.generate_figures              # regenerate reports/figures/
```

Each script writes into `models/` and `reports/metrics/`, matching the
files already committed in this repo (so you can diff your reproduction
against the checked-in results).

## License

MIT — see `LICENSE`.
