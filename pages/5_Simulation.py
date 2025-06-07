import streamlit as st
import json
import requests
import os

st.set_page_config(page_title="Simulation", layout="centered")

# === Chargement des top features ===
file_dir = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))

with open(features_path, "r") as f:
    top_features = json.load(f)

# === Vérification des données ===
if "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire client.")
    st.stop()

initial_input = st.session_state["user_input"]

st.title("🧪 Simulation")
st.write("Modifiez une ou plusieurs variables pour voir l'impact sur la prédiction.")

# === Interface de simulation ===
simu_input = {}

st.subheader("Modifiez les valeurs client :")

for feature in top_features:
    val_init = initial_input.get(feature)

    if feature in ["CODE_GENDER"]:
        simu_input[feature] = st.selectbox("Sexe", ["M", "F"], index=["M", "F"].index(val_init))
    elif feature in ["FLAG_OWN_REALTY", "FLAG_PHONE", "FLAG_OWN_CAR", "FLAG_WORK_PHONE", "FLAG_DOCUMENT_3", "REG_CITY_NOT_WORK_CITY"]:
        simu_input[feature] = st.selectbox(feature, [0, 1], index=[0, 1].index(val_init))
    elif feature == "NAME_FAMILY_STATUS":
        options = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
        simu_input[feature] = st.selectbox("Statut familial", options, index=options.index(val_init))
    elif feature == "NAME_EDUCATION_TYPE":
        options = [
            "Secondary / secondary special", "Higher education", "Incomplete higher",
            "Lower secondary", "Academic degree"
        ]
        simu_input[feature] = st.selectbox("Niveau d'éducation", options, index=options.index(val_init))
    elif feature == "OCCUPATION_TYPE":
        options = [
            "Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff",
            "Accountants", "Medicine staff", "Security staff", "Cooking staff", "Cleaning staff",
            "Private service staff", "Low-skill Laborers", "Secretaries", "Waiters/barmen staff",
            "Realty agents", "IT staff", "HR staff"
        ]
        simu_input[feature] = st.selectbox("Type d'emploi", options, index=options.index(val_init))
    elif feature == "CNT_CHILDREN":
        simu_input[feature] = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=val_init, step=1)
    else:
        simu_input[feature] = st.number_input(f"{feature}", value=float(val_init))

# === Bouton de recalcul ===
if st.button("🔁 Recalculer la prédiction"):
    payload = {"values": [simu_input]}
    try:
        API_URL = "https://projet-7-credit-scoring-api.onrender.com/predict"
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            proba = result["proba"]
            prediction = result["prediction"]
            seuil = result["threshold"]

            st.success("✅ Prédiction recalculée avec succès.")
            label = "Crédit refusé ❌" if prediction == 1 else "Crédit accordé ✅"
            couleur = "red" if prediction == 1 else "green"

            st.markdown(f"### {label}")
            st.metric(label="Nouvelle probabilité d'insolvabilité", value=f"{round(proba * 100, 2)} %", delta=f"Seuil : {round(seuil * 100, 2)} %")

        else:
            st.error("Erreur dans l’API : " + response.text)

    except requests.exceptions.RequestException:
        st.error("❌ Erreur de connexion à l’API.")


