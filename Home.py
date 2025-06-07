import streamlit as st

st.set_page_config(page_title="Accueil - Dashboard Crédit", layout="centered")

st.title("🏡 Bienvenue dans le Dashboard de Crédit Scoring")
st.markdown("""
Ce dashboard vous permet d’évaluer la **solvabilité d’un client** de manière transparente et pédagogique.

### 🧭 Parcours utilisateur :
1. **Formulaire client** : renseignez les principales variables du client.
2. **Résultat du scoring** : visualisez la prédiction et la probabilité.
3. **Explication de la décision** : comprenez ce qui a influencé le score (SHAP).
4. **Comparaison** : situez ce client par rapport aux autres.
5. **Simulation** : testez des modifications pour voir leur impact.

---

**🔄 Navigation via le menu de gauche**

Ce tableau de bord est conçu pour les chargés de relation client souhaitant expliquer les décisions de manière claire, rapide et accessible.
""")
