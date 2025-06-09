import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Explication du score", layout="centered")

# === Vérification des données ===
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord faire une prédiction.")
    st.stop()

result = st.session_state["result"]
user_input = st.session_state["user_input"]

# Vérifie que les SHAP sont bien présents dans le JSON
if "shap_values" not in result:
    st.error("🚫 Les valeurs SHAP n'ont pas été renvoyées par l'API. Veuillez les ajouter.")
    st.stop()

# shap_values est une liste de valeurs et top_features doit être dans les résultats
shap_values = result["shap_values"]
top_features = result.get("top_features", [])

# Vérification cohérence
if len(shap_values) != len(top_features):
    st.error("🚫 Erreur : le nombre de SHAP values ne correspond pas aux top features.")
    st.stop()

# === Titre ===
st.title("🔍 Explication de la prédiction")
st.write("Voici l’impact de chaque variable sur la décision prise pour ce client.")

# === Traitement des SHAP ===
df_shap = pd.DataFrame({
    "Variable": top_features,
    "Valeur saisie": [user_input.get(k, "—") for k in top_features],
    "Impact SHAP": shap_values
}).sort_values("Impact SHAP", key=abs, ascending=False)

# === Affichage du graphique interactif ===
fig = px.bar(
    df_shap.head(10),
    x="Impact SHAP",
    y="Variable",
    orientation='h',
    color="Impact SHAP",
    color_continuous_scale="RdYlGn",
    title="Top variables ayant influencé la décision"
)

st.plotly_chart(fig, use_container_width=True)

# === Tableau explicatif ===
with st.expander("📋 Détails des contributions"):
    st.dataframe(df_shap.style.format({"Impact SHAP": "{:.4f}"}), use_container_width=True)

# === Message d'explication simplifié ===
st.markdown("---")
st.info("Un impact SHAP **positif** pousse vers une prédiction **non solvable**, un impact **négatif** vers **solvable**. Plus la barre est grande, plus l’influence est forte.")
