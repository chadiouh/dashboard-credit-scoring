import streamlit as st
import requests

st.set_page_config(page_title="Dashboard Credit Scoring", layout="wide")

# Design de la page d'accueil
st.title("🏠 Dashboard Credit Scoring")

st.markdown("""
Bienvenue dans le dashboard de scoring de crédit.  
Ce tableau de bord vous permet :
- d’évaluer le risque d’un client,
- de visualiser l’explication du modèle,
- de simuler différents scénarios.

Utilisez le menu de gauche pour naviguer entre les fonctionnalités.
""")

# Test de disponibilité de l’API
API_URL = "https://projet-7-credit-scoring-api.onrender.com/predict"  # Remplace si nécessaire

try:
    # Envoie un test pour voir si l’API répond
    response = requests.get(API_URL)
    if response.status_code == 200:
        st.success("✅ API opérationnelle")
    else:
        st.warning(f"⚠️ API répond mais avec le code : {response.status_code}")
except Exception as e:
    st.error(f"❌ API indisponible ou erreur de connexion\n\n{e}")