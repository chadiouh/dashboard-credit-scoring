# models/shap_global.py

import os
import joblib
import shap
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────── Fichiers requis ──────────────────────
BASE_DIR         = os.path.dirname(__file__)
MODEL_PATH       = os.path.join(BASE_DIR, "model_final.pkl")
PREPROC_PATH     = os.path.join(BASE_DIR, "preprocessor.pkl")
BASELINE_PATH    = os.path.join(BASE_DIR, "baseline_row.json")
TOP_FEATURES_PATH = os.path.join(BASE_DIR, "top_features.json")

# Sorties
PKL_OUTPUT_PATH  = os.path.join(BASE_DIR, "shap_summary_validation.pkl")
PNG_OUTPUT_PATH  = os.path.join(BASE_DIR, "shap_summary_validation.png")

# ────────────────────── Chargement des fichiers ──────────────────────
model = joblib.load(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)

with open(BASELINE_PATH, "r") as f:
    base_row = json.load(f)

with open(TOP_FEATURES_PATH, "r") as f:
    top_features = json.load(f)

# ────────────────────── Préparation des données ──────────────────────
X_base = pd.DataFrame([base_row] * 100)
X_proc = preproc.transform(X_base)

if hasattr(X_proc, "toarray"):
    X_proc = X_proc.toarray()

try:
    feature_names = preproc.get_feature_names_out()
except:
    feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]

# ────────────────────── Calcul SHAP ──────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_proc)
shap_values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
})

# ────────────────────── Tri & sélection des features ──────────────────────
shap_df = shap_df.set_index("feature").reindex(top_features).fillna(0).reset_index()

# ────────────────────── Enregistrement des résultats ──────────────────────
with open(PKL_OUTPUT_PATH, "wb") as f:
    pickle.dump(shap_df, f)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(shap_df["feature"][::-1], shap_df["importance"][::-1])
ax.set_xlabel("Importance moyenne (|SHAP|)")
ax.set_title("Top 15 variables influentes – SHAP global")
ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(PNG_OUTPUT_PATH)

print("✅ Fichiers SHAP générés :")
print(f" - DataFrame : {PKL_OUTPUT_PATH}")
print(f" - Image     : {PNG_OUTPUT_PATH}")
