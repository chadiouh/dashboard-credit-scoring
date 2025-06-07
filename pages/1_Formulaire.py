import streamlit as st
import json
import requests

st.set_page_config(page_title="Formulaire client", layout="centered")
st.title("📋 Saisie du profil client")
st.write("Entrez les valeurs des variables utilisées pour la prédiction.")

# === Chargement des top features ===
with open("models/top_features.json", "r") as f:
    top_features = json.load(f)

# === Interface utilisateur : entrées numériques ===
st.subheader("Variables numériques")
user_input = {}

for feature in top_features:
    if feature == "CNT_CHILDREN":
        user_input[feature] = st.number_input("Nombre d'enfants", min_value=0, step=1, value=0)
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

# === Appel API ===
if st.button("📊 Lancer la prédiction"):
    try:
        payload = {"data": user_input}
        # Remplace l'URL ci-dessous par l'URL exacte de ton API Render
        API_URL = "https://credit-scoring-project-ytl6.onrender.com/predict"
response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prédiction obtenue avec succès.")
            st.session_state["result"] = result
            st.session_state["user_input"] = user_input
            st.switch_page("pages/2_Scoring.py")
        else:
            st.error("Erreur dans l'API : " + response.text)
    except requests.exceptions.RequestException:
        st.error("❌ Impossible de contacter l'API.")
