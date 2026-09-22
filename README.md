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
  features.py                  # derives model inputs (time, amount, distance, risk tier) from a raw request
  api.py                       # Flask inference API (app factory, v1 routes, batch scoring)
tests/
  test_features.py             # unit tests for feature math and decision policy
  test_api.py                  # API tests: auth, validation, derivation, batch, CORS, rate limiting
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
enough to serve predictions. The API is a Flask app factory
(`create_app()` in `src/api.py`); `FRAUD_API_KEY` is only checked when the
factory runs, so importing the module never fails on a misconfigured
environment.

```bash
cp .env.example .env
# edit .env and set a real FRAUD_API_KEY, then:
export $(cat .env | xargs)
python -m src.api
```

For production, run it under gunicorn using the factory directly instead:

```bash
gunicorn --factory "src.api:create_app" --bind 0.0.0.0:5001
```

The API starts on `http://localhost:5001`.

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/v1/health` (alias `/health`) | none | Liveness check, current decision thresholds |
| GET | `/v1/model` | none | Model variant, thresholds, and tuned test metrics |
| GET | `/v1/schema` | none | Every model input: type, required/derivable, allowed categorical values |
| GET | `/openapi.json` | none | OpenAPI 3.0 spec for the API |
| POST | `/v1/score` (alias `/score`) | `X-API-Key` | Score one transaction |
| POST | `/v1/score/batch` | `X-API-Key` | Score up to 100 transactions in one call |

`/v1/score` only strictly requires the raw fields with no formula to derive
them: `amt`, `zip`, `lat`, `long`, `city_pop`, `merch_lat`, `merch_long`,
`category`, `state`, `gender`. Everything else is either derived
server-side (from an optional `timestamp`/`dob`, via `src/features.py`) or
left for the model's trained imputers to fill — check `/v1/schema` for the
full breakdown, and the response's `derived_fields` / `imputed_fields` to
see what happened for a given request.

```bash
curl http://localhost:5001/v1/health

curl -X POST http://localhost:5001/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $FRAUD_API_KEY" \
  -d '{
    "amt": 125.50, "zip": 90210, "lat": 34.0522, "long": -118.2437,
    "city_pop": 500000, "merch_lat": 34.05, "merch_long": -118.25,
    "category": "grocery_pos", "state": "CA", "gender": "F",
    "timestamp": "2026-06-15T14:30:00"
  }'
```

Response includes `fraud_probability`, `decision`
(`approve` / `step_up_verification` / `block`), `risk_band`, `action`,
`derived_fields`, `imputed_fields`, `request_id`, `latency_ms`, and
`top_factors` (the top 5 per-feature contributions to the score, from
XGBoost's `pred_contribs`).

Batch scoring takes the same per-transaction shape under a `transactions`
array and returns per-item results (a bad item doesn't fail the whole
batch):

```bash
curl -X POST http://localhost:5001/v1/score/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $FRAUD_API_KEY" \
  -d '{"transactions": [ { ...same fields as above... } ]}'
```

`MODEL_VARIANT` env var (default `full`) selects which trained variant to
serve: `full`, `full_random`, `no_synthetic`, or `no_ip_risk`. See
`.env.example` for the other runtime settings (`ALLOWED_ORIGINS`,
`RATE_LIMIT_PER_MIN`, `MAX_CONTENT_LENGTH`).

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

## Deploy

The API ships as a Docker image (`Dockerfile`) and deploys to
[Render](https://render.com) via the `render.yaml` blueprint.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/GEEK-IN-GIT/gotcha-fraud-detection)

Clicking that button reads `render.yaml` and provisions a single free-tier
web service that builds `Dockerfile` and health-checks `/v1/health`.

- **API key**: `FRAUD_API_KEY` is set to `generateValue: true` in
  `render.yaml`, so Render generates a random value for you at deploy
  time — it's not something you set yourself. Find it in the Render
  dashboard under your service → **Environment** tab, and use it as the
  `X-API-Key` header when calling the API.
- **Cold starts**: Render's free tier spins the service down when idle.
  The first request after a period of inactivity can take ~30s while it
  wakes back up and gunicorn re-imports pandas/scikit-learn/xgboost and
  reloads the model artifacts; subsequent requests are fast.
- **CORS**: `ALLOWED_ORIGINS` in `render.yaml` is preset to
  `https://geek-in-git.github.io` — update it if you're calling the API
  from a different frontend origin.

To build and run the same image locally:

```bash
docker build -t gotcha-fraud-api .
docker run -p 8080:8080 -e PORT=8080 -e FRAUD_API_KEY=changeme gotcha-fraud-api
curl http://localhost:8080/v1/health
```

## License

MIT — see `LICENSE`.
