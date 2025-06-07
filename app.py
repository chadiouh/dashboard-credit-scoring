import streamlit as st
import json
import requests
import os

# === CONFIGURATION ===
API_URL = "https://projet-7-credit-scoring-api.onrender.com/predict"  # Remplace ici par ton vrai lien API
SEUIL = 0.5  # Seuil de décision métier

# === DÉTECTION DU CHEMIN ABSOLU ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === CHARGEMENT DES VARIABLES ===
with open(os.path.join(BASE_DIR, "top_features.json"), "r") as f:
    top_features = json.load(f)

# === TITRE DE L'INTERFACE ===
st.title("🔎 Prédiction de scoring client")

# === FORMULAIRE UTILISATEUR ===
st.markdown("**Veuillez remplir les variables principales du client :**")
user_input = {}

for feature in top_features:
    user_input[feature] = st.number_input(feature, value=0.0)

# === BOUTON DE PREDICTION ===
if st.button("Lancer la prédiction"):
    try:
        values_list = [user_input[feature] for feature in top_features]
        response = requests.post(API_URL, json={"values": values_list})

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"📊 Probabilité d’insolvabilité : {round(prediction, 4)}")

            # Message explicite selon le seuil
            if prediction >= SEUIL:
                st.error("❌ Crédit refusé (risque élevé)")
            else:
                st.success("✅ Crédit accordé (risque acceptable)")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Erreur lors de la requête : {e}")









