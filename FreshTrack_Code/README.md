# FreshTrack — AI/ML Edition

Full machine-learning upgrade of the FreshTrack food-waste tracker
(Edunet Foundation project). Instead of a hardcoded shelf-life lookup
table, this version trains and uses three real ML models:

| Model | Algorithm | Task |
|---|---|---|
| Shelf-life predictor | `RandomForestRegressor` | Predicts how many days an item will last, given category, storage type, temperature, moisture, and perishability |
| Waste-risk predictor | `RandomForestClassifier` | Predicts the probability an item will be wasted before it's used, learning from a per-household usage-speed feature |
| Waste-risk clustering | `KMeans` (k=3) | Groups items into **High / Medium / Low** waste-risk clusters based on shelf life, moisture, and perishability |

## How the pieces fit together

```
generate_data.py   -> freshtrack_dataset.csv   (3000 synthetic grocery records)
train_models.py    -> models/*.joblib, models/metrics.json, plots/*.png
freshtrack_ml.py    -> interactive app that loads the trained models
```

## Setup

```bash
pip install -r requirements.txt
python generate_data.py     # builds the training dataset
python train_models.py      # trains regression + classification + clustering, saves plots
python freshtrack_ml.py      # run the app
```

## What each script does

**`generate_data.py`** — Builds a synthetic dataset that mirrors what
FreshTrack's real waste log would look like over time: 17 food
categories x 3 storage types, with realistic moisture/perishability
profiles and a simulated household usage-speed pattern that determines
whether each item actually got wasted.

**`train_models.py`** — Trains all three models, evaluates them
(MAE/R² for regression; accuracy/precision/recall/F1 for
classification; silhouette score for clustering), and saves:
- `models/shelf_life_regressor.joblib`
- `models/waste_risk_classifier.joblib`
- `models/cluster_model.joblib`
- `models/metrics.json`
- `plots/clusters.png` — PCA-projected scatter of the 3 waste-risk clusters
- `plots/confusion_matrix.png` — classifier performance
- `plots/feature_importance.png` — which features drive waste risk most

**`freshtrack_ml.py`** — The app. Menu-driven CLI with:
1. Add item -> ML predicts shelf life, waste-risk %, and cluster group
2. Use-It-Soon reminder list, sorted by urgency *and* ML waste-risk score
3. Recipe suggestions per category
4. Mark used / log wasted — every logged item updates this household's
   usage-speed profile, which feeds back into future predictions (this
   is the "waste log improves predictions over time" feature from the
   original report, now actually implemented with a live signal instead
   of a static average)
5. Waste summary
6. Full inventory view
7. ML model report — prints regression/classification metrics and
   cluster stats on demand

## On the image-recognition step

The original report specifies a Teachable Machine (TensorFlow/Keras)
model to identify items from photos. That step is simulated here via
keyword matching (`classify_item()`) so the full pipeline runs without
needing a camera, dataset of food photos, or GPU. The 3 ML models
downstream of it (regression, classification, clustering) are real,
trained on real (synthetic) data — swap `classify_item()` for an actual
loaded Keras model later and everything downstream keeps working
unchanged.
