import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="Explication du score", layout="centered")

# === Vérification des données ===
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord faire une prédiction.")
    st.stop()

result = st.session_state["result"]
user_input = st.session_state["user_input"]

# === Vérification des SHAP values ===
if "shap_values" not in result:
    st.error("🚫 Les valeurs SHAP ne sont pas présentes.")
    st.stop()

shap_values = result["shap_values"]

# === Chargement des top features ===
file_dir = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))
with open(features_path, "r") as f:
    top_features = json.load(f)

# === Vérification cohérence
if len(shap_values) != len(top_features):
    st.error("🚫 Longueur incohérente entre shap_values et top_features.")
    st.stop()

# === Construction du dataframe SHAP
shap_df = pd.DataFrame({
    "Variable": top_features,
    "Valeur saisie": [user_input.get(k, "—") for k in top_features],
    "Impact SHAP": shap_values
}).sort_values("Impact SHAP", key=abs, ascending=False)

# === Limitation de l'affichage à 20 variables
df_display = shap_df.head(15)

# === Affichage
st.title("🔍 Explication de la prédiction")
st.write("Voici l’impact des principales variables sur la décision prise pour ce client.")

fig = px.bar(
    df_display,
    x="Impact SHAP",
    y="Variable",
    orientation='h',
    color="Impact SHAP",
    color_continuous_scale="RdYlGn",
    title="Top 15 variables ayant influencé la décision",
    labels={"Impact SHAP": "Impact SHAP", "Variable": "Variable"}
)
fig.update_layout(yaxis=dict(autorange="reversed"))  # plus haut en haut
st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Détails des contributions (limité à 15 variables)"):
    st.dataframe(df_display.style.format({"Impact SHAP": "{:.4f}"}), use_container_width=True)

st.markdown("---")
st.info("Un impact SHAP **positif** pousse vers une prédiction **non solvable**, un impact **négatif** vers **solvable**. Plus la barre est grande, plus l’influence est forte.")
