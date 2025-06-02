import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Résultat du scoring", layout="centered")

# === Vérification des données ===
if "result" not in st.session_state or "user_input" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord remplir le formulaire dans l'étape précédente.")
    st.stop()

result = st.session_state["result"]
user_input = st.session_state["user_input"]

# === Affichage du titre ===
st.title("📈 Résultat de la prédiction")
st.subheader("Analyse de la solvabilité du client")

# === Interprétation texte ===
proba = result["proba"]
threshold = result["threshold"]
prediction = result["prediction"]

classe = "❌ Crédit refusé (Non solvable)" if prediction == 1 else "✅ Crédit accordé (Solvable)"
couleur = "red" if prediction == 1 else "green"

st.markdown(f"### {classe}")
st.metric(label="Probabilité d'insolvabilité", value=f"{round(proba * 100, 2)} %", delta=f"Seuil = {round(threshold * 100, 2)} %")

# === Jauge visuelle avec Plotly ===
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
st.caption("Les prochaines sections vous permettront de comprendre les raisons de cette décision et de comparer ce profil à d'autres clients.")
