import streamlit as st
import pandas as pd
import requests
import json
import os

st.set_page_config(page_title="Simulation", layout="wide")

st.title("🧪 Simulation de demande de crédit")
st.write("Modifiez les variables pour simuler un nouveau client et voir l'impact sur la prédiction.")

# === Chargement des top features ===
file_dir = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))
data_path = os.path.abspath(os.path.join(file_dir, "..", "data", "application_sample.csv"))

with open(features_path, "r") as f:
    top_features = json.load(f)

# === Chargement d'un client aléatoire comme base
@st.cache_data
def load_sample_row():
    df = pd.read_csv(data_path)
    df = df[top_features]
    sample = df.sample(1, random_state=42).to_dict(orient="records")[0]
    return sample

sample_input = load_sample_row()

# === Interface utilisateur pour modifier les 15 variables
user_input = {}
st.markdown("### 🎛️ Modifiez les variables du client :")
for feature in top_features:
    val = sample_input.get(feature, 0.0)
    if isinstance(val, float):
        user_input[feature] = st.number_input(f"{feature}", value=val, step=0.01, format="%.2f")
    else:
        user_input[feature] = st.number_input(f"{feature}", value=float(val))

# === Choix de l'URL selon environnement
IS_RENDER = os.getenv("RENDER", False)
if IS_RENDER:
    API_URL = "https://api-dashboard-credit-scoring.onrender.com/predict"
else:
    API_URL = "http://127.0.0.1:8000/predict"

# === Appel API
if st.button("🔍 Prédire avec ces valeurs"):
    try:
        ordered_values = [user_input[feat] for feat in top_features]
        payload = {"values": ordered_values}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")
            proba = result.get("proba")

            st.success(f"✅ Prédiction : {'Approuvé' if prediction == 0 else 'Refusé'}")
            st.metric("Probabilité d'insolvabilité", f"{proba*100:.2f} %")

        else:
            st.error(f"Erreur API : {response.text}")
    except Exception as e:
        st.error(f"❌ Erreur de requête : {e}")