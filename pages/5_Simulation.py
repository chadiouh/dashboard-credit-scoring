import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.markdown("# Page de simulation")
st.write("Ajustez les valeurs pour simuler une prédiction.")

# Charger les données originales
df = pd.read_csv("data/application_sample.csv")

# Sélectionner un échantillon
sample = df.sample(1, random_state=42)
sample_dict = sample.to_dict(orient="records")[0]

st.markdown("### Modifier les valeurs pour simulation :")

# Création du formulaire dynamique
user_input = {}
for key, value in sample_dict.items():
    if isinstance(value, (int, float)):
        user_input[key] = st.number_input(f"{key}", value=value)
    else:
        continue  # ne jamais inclure les variables non numériques

# Affichage de la prédiction
if st.button("Prédire avec les nouvelles valeurs"):
    try:
        response = requests.post(
            "https://credit-api-4q4r.onrender.com/predict",
            json={"values": [user_input]}
        )
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            probability = response.json().get("probability")
            st.success(f"Prédiction : {'Approuvé' if prediction==0 else 'Refusé'} (Score : {probability:.2f})")

            # Affichage de la jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Probabilité d'approbation"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig)
        else:
            st.error("Erreur API : " + str(response.text))
    except Exception as e:
        st.error(f"Erreur de requête : {e}")
