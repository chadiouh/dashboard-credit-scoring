import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Résultat du scoring", layout="centered")

# === Vérification des données ===
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire dans l'étape précédente.")
    st.stop()

result = st.session_state["result"]
user_input = st.session_state["user_input"]

# === Variables principales ===
proba = result["proba"]
threshold = result["threshold"]
prediction = result["prediction"]
shap_values = result["shap_values"]
expected_value = result["expected_value"]

# === Titre & décision ===
st.title("📈 Résultat de la prédiction")
st.subheader("Analyse de la solvabilité du client")

classe = "❌ Crédit refusé (Non solvable)" if prediction == 1 else "✅ Crédit accordé (Solvable)"
couleur = "red" if prediction == 1 else "green"

st.markdown(f"### {classe}")
st.metric(label="Probabilité d'insolvabilité", value=f"{round(proba * 100, 2)} %", delta=f"Seuil = {round(threshold * 100, 2)} %")

# === Jauge visuelle Plotly ===
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=proba,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Probabilité d'insolvabilité", 'font': {'size': 22}},
    delta={'reference': threshold},
    gauge={
        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': couleur},
        'steps': [
            {'range': [0, threshold], 'color': 'lightgreen'},
            {'range': [threshold, 1], 'color': 'lightcoral'}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': threshold
        }
    }
))
st.plotly_chart(fig, use_container_width=True)

# === Explication simple ===
st.markdown("---")
st.info("Le score représente la probabilité que le client **ne rembourse pas** son crédit. Une valeur supérieure au seuil entraîne un refus automatique.")

# === SHAP Local Explanation ===
st.subheader("🧠 Contribution des variables à cette décision (SHAP)")

df_shap = pd.DataFrame({
    "feature": top_features,
    "shap_value": shap_values,
    "value": input_values,
    }).sort_values(by="Contribution SHAP", key=abs, ascending=False)

st.dataframe(df_shap.style.format({"Valeur client": "{:.2f}", "Contribution SHAP": "{:.3f}"}))

st.bar_chart(df_shap.set_index("Feature")["Contribution SHAP"])

st.caption("Les valeurs SHAP indiquent l’impact de chaque variable sur la prédiction. Un score négatif tire vers l’acceptation du crédit, un score positif tire vers le refus.")
