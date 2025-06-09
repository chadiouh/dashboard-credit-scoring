import streamlit as st
import json
import requests
import os

st.set_page_config(page_title="Formulaire client", layout="centered")
st.title("📋 Saisie du profil client")
st.write("Entrez les valeurs des variables utilisées pour la prédiction.")

# === Chargement des top features ===
file_dir = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))

with open(features_path, "r") as f:
    top_features = json.load(f)

# === Interface utilisateur : entrées numériques ===
st.subheader("Variables numériques")
user_input = {}

for feature in top_features:
    if feature == "CNT_CHILDREN":
        val = st.number_input("Nombre d'enfants", min_value=0, step=1, value=0)
        user_input[feature] = int(val)
    else:
        val = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.2f")
        user_input[feature] = float(val)

# === Choix de l'URL selon l'environnement
IS_RENDER = os.getenv("RENDER", False)
if IS_RENDER:
    API_URL = "https://projet-7-credit-scoring-api.onrender.com/predict"
else:
    API_URL = "http://127.0.0.1:8000/predict"

# === Appel API ===
if st.button("📊 Lancer la prédiction"):
    try:
        ordered_values = [user_input[feature] for feature in top_features]
        payload = {"values": ordered_values}

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            proba_percent = round(result.get("proba", 0.0) * 100, 2)
            prediction = result.get("prediction")
            threshold = result.get("threshold")

            st.success(f"✅ Probabilité de défaut : {proba_percent} %")

            with st.expander("📌 Détails de la prédiction"):
                st.write(f"**Score brut (proba)** : {proba_percent} %")
                st.write(f"**Seuil de décision** : {threshold}")
                st.write(f"**Décision finale** : {'❌ Défaut' if prediction == 1 else '✅ Approuvé'}")

            # Stockage pour navigation inter-pages
            st.session_state["result"] = result
            st.session_state["user_input"] = user_input

        else:
            st.error("Erreur dans l'API : " + response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de connexion à l’API : {e}")
