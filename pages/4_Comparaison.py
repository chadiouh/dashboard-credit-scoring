import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="Comparaison client", layout="centered")

# === Vérification de session ===
if "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire client.")
    st.stop()

user_input = st.session_state["user_input"]

# === Chargement des top features ===
file_dir = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))
data_path = os.path.abspath(os.path.join(file_dir, "..", "data", "application_sample.csv"))  # <-- correction ici

with open(features_path, "r") as f:
    top_features = json.load(f)

# === Chargement des données de référence ===
@st.cache_data
def load_reference_data():
    df = pd.read_csv(data_path, usecols=top_features)
    df_sample = df.sample(n=10000, random_state=42)  # perf
    return df_sample

df = load_reference_data()

# === Interface ===
st.title("📊 Comparaison du client avec la population")
st.write("Visualisez où se situe ce client par rapport à l'ensemble de la base sur chaque variable.")

feature = st.selectbox("Choisissez une variable :", top_features)

# === Vérification de la variable choisie ===
if feature not in df.columns:
    st.error(f"La variable {feature} n'est pas disponible dans les données de référence.")
    st.stop()

# === Graphique avec valeur client ===
fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution de {feature}")
fig.add_vline(
    x=user_input.get(feature),
    line_dash="dash",
    line_color="red",
    annotation_text="Client",
    annotation_position="top right"
)

st.plotly_chart(fig, use_container_width=True)
st.info(f"Valeur du client pour **{feature}** : `{user_input.get(feature)}`")



