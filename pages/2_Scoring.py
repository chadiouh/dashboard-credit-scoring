import os
import sys
import json
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pickle

st.set_page_config(page_title="Résultat du scoring", layout="centered")

# ──────────────────── Vérification session ────────────────────
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire.")
    st.stop()

result      = st.session_state["result"]
user_input  = st.session_state["user_input"]
proba       = result.get("proba", 0.0)
threshold   = result.get("threshold", 0.5)
prediction  = result.get("prediction", 0)
expected_val = result.get("expected_value")
shap_values = result.get("shap_values", [])

# ──────────────────── Chemins utiles ────────────────────
file_dir     = os.path.dirname(__file__)
models_dir   = os.path.abspath(os.path.join(file_dir, "..", "models"))
features_fp  = os.path.join(models_dir, "top_features.json")
shap_df_fp   = os.path.join(models_dir, "shap_summary_validation.pkl")

with open(features_fp, "r") as f:
    top_features = json.load(f)

# Harmonisation SHAP values
if isinstance(shap_values, list):
    if len(shap_values) == 1 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]
    shap_values = (shap_values + [0.0]*len(top_features))[:len(top_features)]
else:
    shap_values = [float(shap_values)] * len(top_features)

# ──────────────────── JAUGE ────────────────────
st.title("📈 Résultat de la prédiction")

decision_txt = "✅ Crédit accordé (Solvable)" if prediction == 0 else "❌ Crédit refusé (Non solvable)"
decision_col = "green" if prediction == 0 else "red"
st.markdown(f"### {decision_txt}")
st.metric("Probabilité d'insolvabilité", f"{proba*100:.2f} %", delta=f"Seuil : {threshold*100:.2f} %")

gauge = go.Figure(
    go.Indicator(
        mode = "gauge+number+delta",
        value = proba,
        delta = {"reference": threshold},
        gauge = {
            "axis": {"range": [0, 1]},
            "bar": {"color": decision_col},
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

# ──────────────────── SHAP GLOBAL – matplotlib depuis pickle ────────────────────
st.subheader("📊 Importance globale des variables (SHAP)")

try:
    with open(shap_df_fp, "rb") as f:
        df_global = pickle.load(f)

    # s’assurer que seules les top features sont affichées et dans le bon ordre
    df_global = df_global.set_index("feature").reindex(top_features).fillna(0).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df_global["feature"][::-1], df_global["importance"][::-1])
    ax.set_xlabel("Importance moyenne (|SHAP|)")
    ax.set_title("Top 15 variables influentes – SHAP global")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Le fichier 'shap_summary_validation.pkl' est introuvable dans models/")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du SHAP global : {e}")

# ──────────────────── (Optionnel) expected value ────────────────────
if expected_val is not None:
    st.caption(f"Valeur de base du modèle (expected value) : **{expected_val:.4f}**")
