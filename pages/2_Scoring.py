# pages/2_Scoring.py
import os
import sys
import json
import importlib

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Résultat du scoring", layout="centered")

# ──────────────────── Sécurité d’accès ────────────────────
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire.")
    st.stop()

result      = st.session_state["result"]
user_input  = st.session_state["user_input"]

# ──────────────────── Variables principales ────────────────────
proba        = result.get("proba", 0.0)
threshold    = result.get("threshold", 0.5)          # valeur par défaut si non renvoyée
prediction   = result.get("prediction", 0)
shap_values  = result.get("shap_values", [])
expected_val = result.get("expected_value", None)    # peut être None sans gêner la suite

# ──────────────────── Récupération de la liste de features ────────────────────
file_dir      = os.path.dirname(__file__)
features_path = os.path.abspath(os.path.join(file_dir, "..", "models", "top_features.json"))
with open(features_path, "r") as f:
    top_features = json.load(f)          # liste de 15 variables

# Harmonisation éventuelle de la longueur des SHAP
if isinstance(shap_values, list):
    if len(shap_values) == 1 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]
    if len(shap_values) != len(top_features):
        # on tronque ou remplit pour éviter les crash
        shap_values = (shap_values + [0.0]*len(top_features))[:len(top_features)]
else:
    shap_values = [float(shap_values)] * len(top_features)

# ──────────────────── JAUGE + DÉCISION ────────────────────
st.title("📈 Résultat de la prédiction")

decision_txt  = "✅ Crédit accordé (Solvable)" if prediction == 0 else "❌ Crédit refusé (Non solvable)"
decision_col  = "green" if prediction == 0 else "red"
st.markdown(f"### {decision_txt}")
st.metric("Probabilité d'insolvabilité", f"{proba*100:.2f} %", delta=f"Seuil : {threshold*100:.2f} %")

gauge = go.Figure(
    go.Indicator(
        mode  = "gauge+number+delta",
        value = proba,
        delta = {"reference": threshold},
        gauge = {
            "axis": {"range": [0, 1]},
            "bar":  {"color": decision_col},
            "steps": [
                {"range": [0, threshold], "color": "lightgreen"},
                {"range": [threshold, 1], "color": "lightcoral"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold},
        },
        title = {"text": "Probabilité d'insolvabilité", "font": {"size": 22}},
        domain = {"x": [0, 1], "y": [0, 1]},
    )
)
st.plotly_chart(gauge, use_container_width=True)

st.markdown("---")
st.info("Le score représente la probabilité que le client **ne rembourse pas** son crédit. "
        "Une valeur supérieure au seuil entraîne un refus automatique.")

# ──────────────────── SHAP GLOBAL (15 variables) ────────────────────
st.subheader("📊 Importance globale des variables (SHAP)")

try:
    # import dynamique du script de calcul (models/shap_plot.py)
    models_dir = os.path.abspath(os.path.join(file_dir, "..", "models"))
    if models_dir not in sys.path:
        sys.path.append(models_dir)

    shap_plot = importlib.import_module("shap_plot")            # exécute le code une 1ʳᵉ fois
    df_global = getattr(shap_plot, "shap_df", None) or shap_plot.get_shap_df()

    # on garde exactement les 15 features du dashboard, dans l’ordre défini
    df_global = df_global.set_index("feature").reindex(top_features).reset_index()
    df_global["importance"].fillna(0.0, inplace=True)

    # tracé matplotlib (15 barres ≡ 15 features)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df_global["feature"][::-1], df_global["importance"][::-1])
    ax.set_xlabel("Importance moyenne (|SHAP|)")
    ax.set_title("Top 15 variables influentes – SHAP global")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Impossible d’afficher le graphique global : {e}")

# On peut ajouter un petit rappel de la valeur de base si on l’a :
if expected_val is not None:
    st.caption(f"Valeur SHAP de base (expected value) : **{expected_val:.4f}**")