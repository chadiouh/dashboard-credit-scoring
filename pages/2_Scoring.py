import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os

st.set_page_config(page_title="Résultat du scoring", layout="centered")

# ──────────────────── Vérifications ────────────────────
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire.")
    st.stop()

result      = st.session_state["result"]
user_input  = st.session_state["user_input"]

# ──────────────────── Variables principales ────────────────────
proba        = result["proba"]
threshold    = result["threshold"]
prediction   = result["prediction"]
shap_values  = result["shap_values"]
expected_val = result["expected_value"]

# ──────────────────── Récupération des 15 features cibles ────────────────────
file_dir      = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))
with open(features_path, "r") as f:
    top_features = json.load(f)                  # -> liste des 15 variables

input_values = [user_input.get(feat, "—") for feat in top_features]

# ─── Harmonisation de la longueur des SHAP ───
if isinstance(shap_values, list):
    # s'il reste un niveau de liste (ex. [[...]]), on l'aplatit
    if len(shap_values) == 1 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]

    # tronquage / remplissage
    if len(shap_values) > len(top_features):
        shap_values = shap_values[:len(top_features)]
    elif len(shap_values) < len(top_features):
        shap_values = shap_values + [0.0]*(len(top_features) - len(shap_values))
else:
    # valeur scalaire (peu probable) → réplication
    shap_values = [float(shap_values)]*len(top_features)

# ──────────────────── En-tête & jauge ────────────────────
st.title("📈 Résultat de la prédiction")
st.subheader("Analyse de la solvabilité du client")

classe  = "❌ Crédit refusé (Non solvable)" if prediction else "✅ Crédit accordé (Solvable)"
couleur = "red" if prediction else "green"

st.markdown(f"### {classe}")
st.metric("Probabilité d'insolvabilité", f"{proba*100:.2f} %", delta=f"Seuil = {threshold*100:.2f} %")

fig = go.Figure(go.Indicator(
    mode   = "gauge+number+delta",
    value  = proba,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title  = {'text': "Probabilité d'insolvabilité", 'font': {'size': 22}},
    delta  = {'reference': threshold},
    gauge  = {
        'axis': {'range': [0, 1]},
        'bar':  {'color': couleur},
        'steps': [
            {'range': [0, threshold], 'color': 'lightgreen'},
            {'range': [threshold, 1], 'color': 'lightcoral'}
        ],
        'threshold': {'line': {'color': "black", 'width': 4}, 'value': threshold}
    }
))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("Le score représente la probabilité que le client **ne rembourse pas** son crédit. "
        "Une valeur supérieure au seuil entraîne un refus automatique.")

# ──────────────────── SHAP global  ────────────────────
# ──────────────────── SHAP global dynamique ────────────────────
st.subheader("📊 Importance globale des variables (SHAP)")

try:
    import importlib
    import matplotlib.pyplot as plt
    import sys, os

    # Pour pouvoir importer le script situé dans /models
    models_dir = os.path.abspath(os.path.join(file_dir, "..", "models"))
    if models_dir not in sys.path:
        sys.path.append(models_dir)

    shap_plot = importlib.import_module("shap_plot")  # exécute le script
    df_shap = getattr(shap_plot, "shap_df", None) or shap_plot.get_shap_df()

    # Barplot matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_shap["feature"][::-1], df_shap["importance"][::-1])
    ax.set_xlabel("Importance moyenne (|SHAP|)")
    ax.set_title("Top variables influentes (SHAP)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Impossible d’afficher le graphique SHAP dynamique : {e}")
