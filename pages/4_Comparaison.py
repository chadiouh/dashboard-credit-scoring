import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="Comparaison client", layout="centered")

# === Vérification des données ===
if "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire client.")
    st.stop()

user_input = st.session_state["user_input"]

# === Chargement des données de référence ===
@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/application_train.csv", usecols=top_features)
    df_sample = df.sample(n=10000, random_state=42)  # échantillon pour perf
    return df_sample


df = load_reference_data()

# === Chargement des top features ===
with open("models/top_features.json", "r") as f:
    top_features = json.load(f)

st.title("📊 Comparaison du client avec la population")
st.write("Visualisez où se situe ce client par rapport à l'ensemble de la base sur chaque variable.")

# === Sélecteur de variable à comparer ===
feature = st.selectbox("Choisissez une variable :", top_features)

# === Vérification de la variable choisie ===
if feature not in df.columns:
    st.error(f"La variable {feature} n'est pas disponible dans les données de référence.")
    st.stop()

# === Création du graphique ===
fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution de {feature} dans la population")
fig.add_vline(x=user_input.get(feature, None), line_dash="dash", line_color="red", annotation_text="Client", annotation_position="top right")

st.plotly_chart(fig, use_container_width=True)

# === Commentaire dynamique ===
val = user_input.get(feature)
st.info(f"Valeur du client pour **{feature}** : `{val}`")
