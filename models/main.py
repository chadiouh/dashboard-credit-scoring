import os
import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# === Chemin vers les fichiers ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Chargement des fichiers ===
with open(os.path.join(BASE_DIR, "top_features.json"), "r") as f:
    top_features = json.load(f)

with open(os.path.join(BASE_DIR, "baseline_row.json"), "r") as f:
    baseline_row = json.load(f)

with open(os.path.join(BASE_DIR, "preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_final.pkl"), "rb") as f:
    model = pickle.load(f)

# === Initialisation de l'API ===
app = FastAPI()

# === Modèle d'entrée ===
class InputData(BaseModel):
    values: List[float]

@app.get("/")
def read_root():
    return {"message": "API de scoring opérationnelle."}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        if len(input_data.values) != len(top_features):
            raise HTTPException(
                status_code=400,
                detail=f"Attendu {len(top_features)} valeurs, reçu {len(input_data.values)}"
            )

        # Insertion des valeurs dans la baseline
        full_input = baseline_row.copy()
        for i, feature in enumerate(top_features):
            full_input[feature] = input_data.values[i]

        # DataFrame + préprocessing
        X = pd.DataFrame([full_input])
        X_processed = preprocessor.transform(X)

        # Prédiction
        proba = model.predict_proba(X_processed)[0, 1]
        threshold = 0.5
        prediction = int(proba > threshold)

        return {
            "prediction": prediction,
            "proba": float(proba),
            "threshold": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
