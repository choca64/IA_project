# Home.py (ou main.py)
import streamlit as st

st.set_page_config(page_title="Accueil", layout="centered")

st.title("🌾 Application de Prédiction Agricole avec un modèle LSTM 🌾")

st.markdown("""
Bienvenue dans l'application intelligente de prédiction de l'état du sol.

Utilisez le menu de gauche pour :
- 🔧 **Entraîner un modèle IA** avec vos données
- 🔮 **Prédire l'état du sol pour une année future** avec ou sans météo réelle

---

Cette plateforme combine météo, intelligence artificielle et science des sols pour vous aider à anticiper l'avenir agricole.
""")

