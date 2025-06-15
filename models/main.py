import json
import os
import joblib
import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ──────────────── Config & chargement ──────────────── #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "top_features.json")) as f:
    top_features = json.load(f)

with open(os.path.join(BASE_DIR, "baseline_row.json")) as f:
    baseline_row = json.load(f)

preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "model_final.pkl"))

# ✅ Chemin corrigé vers X_valid.csv
X = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "X_valid.csv"))

# ──────────────── API Setup ──────────────── #

app = FastAPI()

class InputData(BaseModel):
    values: List

@app.get("/")
def root():
    return {"message": "pong"}

@app.post("/predict/")
def predict(input_data: InputData):
    try:
        # Baseline sur la première ligne de X_valid
        baseline_row = X.iloc[0].copy()
        row = baseline_row.copy()

        # Met à jour les top features avec les valeurs saisies
        for i, feat in enumerate(top_features):
            row[feat] = input_data.values[i]

        row_df = pd.DataFrame([row])

        # Imputation
        row_imputed = preprocessor.transform(row_df)

        # Prédiction
        prediction = model.predict(row_imputed)[0]
        proba = model.predict_proba(row_imputed)[0][1]

        # ───── Partie SHAP locale corrigée ───── #
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_imputed)

        all_feature_names = X.columns.tolist()
        if isinstance(shap_values, list):
            shap_array = shap_values[1]  # classe 1 = défaut
        else:
            shap_array = shap_values

        shap_df = pd.DataFrame(shap_array, columns=all_feature_names)
        shap_vals = shap_df[top_features].iloc[0].tolist()
        # ─────────────────────────────────────── #

        return {
            "prediction": int(prediction),
            "proba": float(proba),
            "shap_values": shap_vals
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
