import streamlit as st
import json
import requests

st.set_page_config(page_title="Formulaire client", layout="centered")

st.title("📋 Saisie du profil client")
st.write("Renseignez les 15 variables clés pour obtenir une prédiction de solvabilité.")

# === Chargement des top features ===
with open("models/top_features.json", "r") as f:
    top_features = json.load(f)

# === Interface utilisateur personnalisée ===
user_input = {}

st.subheader("Informations client")
for feature in top_features:
    if feature in ["CODE_GENDER"]:
        user_input[feature] = st.selectbox("Sexe", ["M", "F"])
    elif feature in ["FLAG_OWN_REALTY", "FLAG_PHONE", "FLAG_OWN_CAR", "FLAG_WORK_PHONE", "FLAG_DOCUMENT_3", "REG_CITY_NOT_WORK_CITY"]:
        user_input[feature] = st.selectbox(feature, [0, 1])
    elif feature in ["NAME_FAMILY_STATUS"]:
        user_input[feature] = st.selectbox("Statut familial", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
    elif feature in ["NAME_EDUCATION_TYPE"]:
        user_input[feature] = st.selectbox("Niveau d'éducation", [
            "Secondary / secondary special", "Higher education", "Incomplete higher",
            "Lower secondary", "Academic degree"
        ])
    elif feature in ["OCCUPATION_TYPE"]:
        user_input[feature] = st.selectbox("Type d'emploi", [
            "Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff",
            "Accountants", "Medicine staff", "Security staff", "Cooking staff", "Cleaning staff",
            "Private service staff", "Low-skill Laborers", "Secretaries", "Waiters/barmen staff",
            "Realty agents", "IT staff", "HR staff"
        ])
    elif feature == "CNT_CHILDREN":
        user_input[feature] = st.number_input("Nombre d'enfants", min_value=0, max_value=10, step=1)
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

# === Appel API ===
if st.button("📊 Lancer la prédiction"):
    try:
        payload = {"data": user_input}
        response = requests.post("https://credit-scoring-project-ytl6.onrender.com", json=payload)

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
