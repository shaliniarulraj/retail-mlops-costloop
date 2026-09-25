# Retail Demand Forecasting — Cost-Aware, Self-Correcting MLOps Pipeline

A closed-loop MLOps system that forecasts retail demand, converts prediction
error into simulated business cost (stockout + holding cost), automatically
retrains when that cost crosses a threshold, and learns from store-manager
overrides via a human-in-the-loop feedback loop.

Every script below has been run and verified working in this exact repo —
this README walks through them in the order you should run them.

## 0. Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 recommended. All package versions are pinned in `requirements.txt`
to versions tested against this codebase.

## 1. Generate data — synthetic, real Rossmann, OR real M5

**This project currently ships with real M5 data as its checked-in state**
(see `git log` — three real commits: synthetic → real Rossmann → real M5).
All three options below are fully working; switch between them anytime by
re-running the relevant loader.

**Option A — synthetic (fast iteration, multi-SKU story, no real-world caveats):**

```bash
python src/data/generate_synthetic_data.py
```

**Option B — real Rossmann, WITH real day-level weather:**

```bash
python src/data/load_real_rossmann.py --weather-csv data/raw/rossmann_weather_raw.csv
```

1,115 real stores, 2013-01-01 to 2015-07-31, 844,392 rows. Joins a real
store→German-state mapping (`rossmann_store_states_raw.csv`) and merges
**real day-level weather** (`rossmann_weather_raw.csv` — the genuine
fast.ai/Kaggle bundle, 16 states, temps from -13°C to 31°C, zero missing
values). If you ever lose that weather file, omit `--weather-csv` and it
falls back to real published regional climate normals instead — still
real climate data, just normals rather than daily observations.

Verified result: adding real weather improved the model over the
normals-only version (cost/row dropped from ₹9,148.83 → ₹9,136.51) and the
promotion gate correctly promoted it on that basis — a clean, citable
result for your report.

**Option C — real M5 (true per-SKU granularity — the current checked-in state):**

```bash
python src/data/load_real_m5.py \
    --calendar-csv /mnt/user-data/uploads/calendar.csv \
    --sales-csv /mnt/user-data/uploads/sales_train_evaluation.csv \
    --n-items 600
```

544 real SKUs across all 10 real M5 stores (CA_1-4, TX_1-3, WI_1-3) and
all 3 categories (FOODS, HOBBIES, HOUSEHOLD) — a random sample, not
`head(n)`, since the file is sorted store-by-store and `head()` would
silently grab only one store. `is_special_day` combines real US calendar
events with each row's **own state's** SNAP (food-assistance) flag —
note this required a real bug fix during development: an early version
OR'd all three states' SNAP flags together regardless of which state the
row belonged to, inflating the special-day rate to an implausible 53%;
the corrected, state-aware version reads 38.3%, which lines up with the
known fact that SNAP covers the first 10 days of each month (~33%
baseline) plus extra event days.

**`--n-items 0` means "no limit"** (all 30,490 series) — be aware this
needs several GB of RAM to melt to long format; this dev sandbox (3.9GB,
1 CPU) OOM'd on the first attempt. Run the full dataset on your own
machine if you want it; 600 series is enough to demonstrate the
per-SKU/multi-store/multi-category story for a course project.

**An honest, important finding from running this:** WMAPE on real M5
data is dramatically worse than on Rossmann (73.7% vs. 9.6%). This is
not a bug — individual per-SKU daily sales are small, sparse, intermittent
numbers, which is inherently much harder to forecast than store-level
totals that average out that noise. State this explicitly in your
report: **per-SKU granularity is a genuinely harder ML problem, not just
an engineering scaling exercise** — and it's consistent with the real M5
competition literature (top solutions report WRMSSE around 0.5–0.6, a
very different metric scale than simple WMAPE on aggregates).

**Five honest things to state in your report regardless of which real
dataset you use:**

1. **Rossmann has no SKU-level granularity** (`sku_id = "SKU_TOTAL"`) —
   describe Rossmann runs as a store-level validation of the pipeline,
   and M5 for the genuine multi-SKU story.
2. **M5 has no real weather** (same limitation Rossmann had before the
   real weather file was supplied) — uses the flat placeholder `18.0`.
   If you want real weather for M5 too, the same climate-normals pattern
   from `load_real_rossmann.py` could be ported across US states.
3. **`is_special_day` (renamed from an earlier `is_festival`)** means a
   different real-world thing depending on the data source: Indian
   festivals (synthetic), German state/school holidays (Rossmann), or US
   holidays + SNAP days (M5). All three represent the same underlying
   modeling concept — "an unusual, calendar-driven demand day" — through
   different real calendars. Don't claim they're literally the same
   event; do claim the *feature generalizes* across calendar systems,
   which is a more interesting and defensible result than pretending
   otherwise.
4. **Lag features are computed over *open* days, not strict calendar
   days** for Rossmann, since closed-store days are dropped before
   feature engineering.
5. **Don't run the cost-based promotion gate across a dataset swap** —
   comparing M5's per-SKU cost scale (₹13/row) against Rossmann's
   per-store-total scale (₹9,136/row) is comparing different units of
   analysis. Promote explicitly when switching datasets (see how each
   swap was promoted manually in the project's development history);
   let the gate compare consecutive retrains *within* the same dataset.

**To swap in your own POS export instead:** write a loader following the
same pattern — map your columns to
`date, store_id, sku_id, sales, is_special_day, temp_c` and everything
downstream works unchanged.

## 2. Track data with DVC

```bash
git init          # if not already a repo
dvc init
dvc add data/raw/sales.csv
git add data/raw/sales.csv.dvc data/raw/.gitignore .dvc
git commit -m "Track raw sales data with DVC"
```

This is what lets you say "model version 3 was trained on data version X" —
add a remote (`dvc remote add -d storage <url>`) once you have cloud storage
to push to.

## 3. Validate the data

```bash
python src/data/validate.py
```

Runs a `pandera` schema check (column types, value ranges, regex patterns
on IDs). **This should be the first step in every pipeline run** — fail
fast here, not three steps later inside a confusing model error.

## 4. Build features

```bash
python src/features/build_features.py
```

**Output:** `data/processed/features.parquet`

Adds: day-of-week/weekend/month/day-of-year, lag features (1/7/14/28 days),
rolling mean+std (7/28 day windows), and a `festival_in_next_3d` flag (so
the model can react to an upcoming festival, not just one that already
happened).

## 5. Train the model (with MLflow tracking)

```bash
python src/training/train.py
```

Trains a LightGBM model (Tweedie objective — handles the zero-inflated,
intermittent demand you'll see in real retail data better than plain
regression), does a time-based train/validation split, and logs to MLflow:

- Standard metrics: MAE, WMAPE
- **Business metrics: total/avg simulated cost, broken into stockout vs.
  holding cost** — this is the metric that matters for your novelty claim

View everything in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```

## 6. The cost calculator (run its own demo)

```bash
python src/cost/cost_calculator.py
```

This is the core novel module — `src/cost/cost_calculator.py`. It has no
external dependency beyond numpy/pandas; everything else in the pipeline
calls into it. Run the unit tests to see it's correctness-checked:

```bash
pytest tests/ -v
```

## 7. Cost-based retrain trigger

```bash
python src/pipeline/retrain_trigger.py
```

Pulls the baseline cost **dynamically from whichever model currently holds
the `production` alias in MLflow** (not a hardcoded number — this matters
because the baseline is wildly different between synthetic SKU-level data
and real store-level data, and a hardcoded constant would silently go
stale the moment you swap datasets). Takes the most recent 14 days of
data, **injects a synthetic demand shock** into one store (simulating an
unmodeled local event), and checks whether the resulting cost-per-row
exceeds `1.5×` that baseline. If it does, it prints
`>>> RETRAIN TRIGGERED <<<` and exits with code `2` — deliberately, so a
CI/CD pipeline can branch on that exit code without parsing log text.

It also prints a per-store cost breakdown, so you can see *which* store is
costing you money, not just that something drifted.

## 8. Promote-if-better gate

```bash
python src/pipeline/promote_if_better.py
```

Compares the newest MLflow-registered model version's `business_cost_avg_per_row`
against whatever model currently holds the `production` alias. Only
re-assigns the alias if the new model is actually cheaper to run. This is
your CI/CD deploy gate — retraining doesn't mean redeploying.

## 9. Serve the model

```bash
uvicorn src.serving.app:app --reload --port 8000
```

Endpoints:
- `GET /health` — readiness check
- `POST /predict` — raw demand forecast
- `POST /recommend` — forecast + safety stock → recommended order quantity
- `GET /metrics` — Prometheus-format metrics (latency, request counts)

Try it:

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{
  "store_id": "STORE_1", "sku_id": "SKU_0002",
  "is_festival": 1, "temp_c": 28.0, "day_of_week": 2, "is_weekend": 0,
  "month": 11, "day_of_year": 305,
  "sales_lag_1": 40, "sales_lag_7": 38, "sales_lag_14": 35, "sales_lag_28": 33,
  "sales_rolling_mean_7": 37.5, "sales_rolling_std_7": 4.2,
  "sales_rolling_mean_28": 35.0, "sales_rolling_std_28": 5.1,
  "festival_in_next_3d": 1
}'
```

## 10. The human-in-the-loop feedback console

```bash
streamlit run src/feedback/override_app.py
```

Opens a simple "store manager" UI: pick a store/SKU, see the system's
recommendation, optionally override it with a reason, submit. Every
override is logged to `data/processed/feedback.db` (SQLite) with both the
system's number and the manager's number.

## 11. Fold overrides back into training (closing the loop)

```bash
python src/feedback/fold_into_training.py
```

Pulls every override not yet used in training, replaces that row's label
in `features.parquet` with the manager's corrected number, and marks it
`folded_into_training = 1` in the database. **Run this before
`train.py` whenever you retrain** — it's what makes the human correction
actually change the model, not just sit in a log table.

## 12. Drift detection

```bash
python src/monitoring/drift_check.py
```

Compares the training-time feature distribution to a recent (synthetically
shocked) batch using Evidently AI, and saves a full visual report to
`data/processed/drift_report.html` — open it in a browser.

## 13. Run it all in Docker

```bash
cd deployment
docker compose up --build
```

Brings up:
- the FastAPI app on `localhost:8000`
- Prometheus on `localhost:9090` (scraping the app's `/metrics`)
- Grafana on `localhost:3000` (default login `admin` / `admin`) — point it
  at the Prometheus data source to build dashboards

## 14. CI/CD

`.github/workflows/ci.yml` wires phases 3–11 into one pipeline:

```
test → validate+build features → fold in overrides
     → cost-based retrain check
         → (if triggered) retrain → promote-if-better gate
     → build Docker image
```

It also runs on a daily cron schedule, not just on push — because a
demand shock can happen on a day nobody touches the code.

## Project structure

```
retail-mlops-project/
├── data/
│   ├── raw/sales.csv              # DVC-tracked
│   └── processed/                 # features, model, feedback.db, drift_report.html
├── src/
│   ├── data/                      # generation + validation
│   ├── features/                  # feature engineering
│   ├── training/                  # MLflow-tracked training
│   ├── cost/                      # *** the novelty: cost_calculator.py ***
│   ├── pipeline/                  # retrain_trigger.py, promote_if_better.py
│   ├── serving/                   # FastAPI app
│   ├── feedback/                  # *** the novelty: human-in-the-loop loop ***
│   └── monitoring/                # Evidently drift checks
├── tests/                         # pytest unit tests
├── deployment/                    # Dockerfile, docker-compose.yml, prometheus.yml
└── .github/workflows/ci.yml
```

## What's intentionally simplified for course scope

- **One global cost rate** (₹25 stockout / ₹4 holding per unit) instead of
  per-SKU/category rates — swap `DEFAULT_STOCKOUT_COST_PER_UNIT` and
  `DEFAULT_HOLDING_COST_PER_UNIT` in `src/cost/cost_calculator.py` once you
  have real estimates.
- **Synthetic overrides** instead of real store-manager data — be upfront
  about this in your report (see the earlier conversation about this
  trade-off).
- **A single baseline model** (LightGBM, no per-series hierarchical
  reconciliation) — deliberately, since we scoped that pillar out in favor
  of the feedback loop.
- **`promote_if_better.py` runs as a script**, not a fully automated
  Step-Functions-style state machine — good enough for course scope; AWS's
  SageMaker Pipelines pattern is the production-grade version of this same
  idea (see the earlier "existing systems" comparison).

## A genuinely honest result from building this

When this repo's gate was tested with three candidate retrains, two more
complex configurations both scored *worse* on cost than the simple baseline
and were correctly rejected. That's not a bug — it's the gate doing its
job. Don't be surprised (or worried) if your real results look similar;
write it up as evidence the cost gate works, not as a failure.
