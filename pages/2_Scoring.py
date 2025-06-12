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
st.subheader("📊 Importance globale des variables (SHAP)")

try:
    import shap
    import pickle
    import numpy as np
    import plotly.express as px

    # === Chemins
    model_path = os.path.join(file_dir, "..", "models", "model_final.pkl")
    preproc_path = os.path.join(file_dir, "..", "models", "preprocessor.pkl")
    baseline_path = os.path.join(file_dir, "..", "models", "baseline_row.json")
    top_feat_path = os.path.join(file_dir, "..", "models", "top_features.json")

    # === Chargement
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preproc_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(baseline_path, "r") as f:
        baseline_row = json.load(f)
    with open(top_feat_path, "r") as f:
        top_features = json.load(f)

    # === Création d’un DataFrame fictif cohérent
    X_full = pd.DataFrame([baseline_row] * 100)

    # === Transformation
    X_proc = preprocessor.transform(X_full)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    # === SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_proc)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]

    # === Moyenne sur l’ensemble
    mean_shap = np.abs(shap_vals).mean(axis=0)

    # === Récupération des noms après transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"Var_{i}" for i in range(X_proc.shape[1])]

    # === Création du dataframe SHAP global
    df_all = pd.DataFrame({
        "Variable": feature_names,
        "Importance SHAP moyenne": mean_shap
    })

    # === Filtrage pour ne garder que les top features
    df_top = df_all[df_all["Variable"].isin(top_features)].copy()
    df_top = df_top.sort_values("Importance SHAP moyenne", ascending=True)

    # === Affichage
    fig = px.bar(
        df_top,
        x="Importance SHAP moyenne",
        y="Variable",
        orientation="h",
        color="Importance SHAP moyenne",
        color_continuous_scale="Bluered_r",
        title="Top 15 variables utilisées dans le dashboard"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Graphique basé sur un profil type, recalculé dynamiquement avec SHAP sur les 15 variables utilisées dans l’interface.")

except Exception as e:
    st.error(f"❌ Erreur lors du calcul du SHAP global : {e}")
