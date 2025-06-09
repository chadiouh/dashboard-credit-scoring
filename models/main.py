import os
import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import shap

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "top_features.json")) as f:
    top_features = json.load(f)

with open(os.path.join(BASE_DIR, "baseline_row.json")) as f:
    baseline_row = json.load(f)

with open(os.path.join(BASE_DIR, "preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_final.pkl"), "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    values: List[float]

@app.get("/")
def root():
    return {"message": "API active"}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Suppression de la vérification stricte du nombre de variables
        # if len(input_data.values) != len(top_features):
        #     raise HTTPException(status_code=400, detail="Mauvais nombre de variables")

        # Création du dataframe
        row = baseline_row.copy()
        for i, feat in enumerate(top_features):
            row[feat] = input_data.values[i]
        X = pd.DataFrame([row])
        X_processed = preprocessor.transform(X)

        # Prédiction
        proba = model.predict_proba(X_processed)[0, 1]
        prediction = int(proba >= 0.5)

        # Explication SHAP (TreeExplainer)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            raw_values = shap_values[1][0]
            shap_vals = raw_values[:len(top_features)].tolist()
            expected_val = float(explainer.expected_value[1])
        else:
            raw_values = shap_values[0]
            shap_vals = raw_values[:len(top_features)].tolist()
            expected_val = float(explainer.expected_value)

        return {
            "prediction": prediction,
            "proba": proba,
            "threshold": 0.5,
            "shap_values": shap_vals,
            "expected_value": expected_val
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
