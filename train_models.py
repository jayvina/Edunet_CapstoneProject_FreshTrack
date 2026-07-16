"""
train_models.py
================
Trains the three ML components of FreshTrack:

  1. REGRESSION      RandomForestRegressor  -> predicts shelf_life_days
  2. CLASSIFICATION  RandomForestClassifier -> predicts waste risk (P(wasted))
  3. CLUSTERING      KMeans (k=3)           -> groups items into
                                                waste-risk clusters
                                                (High / Medium / Low risk)

Saves trained models + evaluation metrics + plots so freshtrack_ml.py
(the app) can load them instantly without retraining.
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

import os
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("freshtrack_dataset.csv")

CATEGORICAL = ["category", "storage_type"]
NUMERIC = ["quantity", "purchase_temperature_c", "moisture_level", "perishability_index", "household_usage_speed"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

X = df[CATEGORICAL + NUMERIC]

# -----------------------------------------------------------------------
# 1. REGRESSION — predict shelf_life_days
# -----------------------------------------------------------------------
y_reg = df["shelf_life_days"]
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)),
])
reg_pipeline.fit(X_train, y_train)
y_pred = reg_pipeline.predict(X_test)

reg_metrics = {
    "mae_days": round(mean_absolute_error(y_test, y_pred), 2),
    "r2_score": round(r2_score(y_test, y_pred), 3),
}
joblib.dump(reg_pipeline, "models/shelf_life_regressor.joblib")

# -----------------------------------------------------------------------
# 2. CLASSIFICATION — predict wasted (0/1) -> waste risk probability
# -----------------------------------------------------------------------
y_clf = df["wasted"]
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

clf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")),
])
clf_pipeline.fit(X_train, y_train)
y_pred_clf = clf_pipeline.predict(X_test)

clf_metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred_clf), 3),
    "precision": round(precision_score(y_test, y_pred_clf), 3),
    "recall": round(recall_score(y_test, y_pred_clf), 3),
    "f1_score": round(f1_score(y_test, y_pred_clf), 3),
}
joblib.dump(clf_pipeline, "models/waste_risk_classifier.joblib")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_clf, display_labels=["Not wasted", "Wasted"])
plt.title("Waste Risk Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance for the classifier (which factors drive waste risk)
feature_names = (
    list(clf_pipeline.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    + NUMERIC
)
importances = clf_pipeline.named_steps["model"].feature_importances_
top_idx = np.argsort(importances)[-10:]
plt.figure(figsize=(8, 5))
plt.barh([feature_names[i] for i in top_idx], importances[top_idx], color="#2e7d32")
plt.xlabel("Importance")
plt.title("Top Features Driving Waste Risk")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 3. CLUSTERING — KMeans groups items into waste-risk clusters
# -----------------------------------------------------------------------
cluster_features = df[["shelf_life_days", "moisture_level", "perishability_index"]]
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(cluster_scaled)
sil_score = silhouette_score(cluster_scaled, cluster_labels)

# Name clusters by their mean shelf life: shortest life = highest risk
df["cluster"] = cluster_labels
cluster_order = df.groupby("cluster")["shelf_life_days"].mean().sort_values().index.tolist()
risk_names = ["High risk (use within days)", "Medium risk", "Low risk (long shelf life)"]
cluster_name_map = {cluster_id: risk_names[rank] for rank, cluster_id in enumerate(cluster_order)}

joblib.dump({"kmeans": kmeans, "scaler": scaler, "cluster_name_map": cluster_name_map}, "models/cluster_model.joblib")

# 2D PCA visualization of the clusters
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(cluster_scaled)
plt.figure(figsize=(7, 6))
colors = {0: "#c62828", 1: "#f9a825", 2: "#2e7d32"}
for cluster_id in sorted(df["cluster"].unique()):
    mask = df["cluster"] == cluster_id
    plt.scatter(
        coords[mask, 0], coords[mask, 1],
        s=10, alpha=0.5,
        label=cluster_name_map[cluster_id],
        color=colors.get(cluster_id % 3),
    )
plt.legend()
plt.title("Food Item Waste-Risk Clusters (PCA projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig("plots/clusters.png", dpi=150)
plt.close()

cluster_summary = (
    df.groupby("cluster")
    .agg(
        avg_shelf_life=("shelf_life_days", "mean"),
        avg_moisture=("moisture_level", "mean"),
        avg_perishability=("perishability_index", "mean"),
        waste_rate=("wasted", "mean"),
        count=("wasted", "size"),
    )
    .round(2)
)
cluster_summary["risk_label"] = cluster_summary.index.map(cluster_name_map)

# -----------------------------------------------------------------------
# Save metrics for the app to display
# -----------------------------------------------------------------------
metrics = {
    "regression": reg_metrics,
    "classification": clf_metrics,
    "clustering": {
        "silhouette_score": round(sil_score, 3),
        "clusters": json.loads(cluster_summary.to_json(orient="index")),
    },
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("\n[Regression] Shelf-life prediction (RandomForestRegressor)")
print(f"  MAE:  {reg_metrics['mae_days']} days")
print(f"  R^2:  {reg_metrics['r2_score']}")

print("\n[Classification] Waste risk prediction (RandomForestClassifier)")
for k, v in clf_metrics.items():
    print(f"  {k}: {v}")

print("\n[Clustering] Waste-risk groups (KMeans, k=3)")
print(f"  Silhouette score: {round(sil_score, 3)}")
print(cluster_summary[["risk_label", "avg_shelf_life", "waste_rate", "count"]])

print("\nSaved models -> models/")
print("Saved plots  -> plots/ (clusters.png, confusion_matrix.png, feature_importance.png)")
