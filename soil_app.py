import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Charger les prédictions ===
st.set_page_config(page_title="Prédiction de la qualité du sol", layout="wide")
st.title("🌾 Estimation de la qualité du sol - Prédiction IA (2024)")

@st.cache_data
def load_data():
    df = pd.read_csv("soil_multi_prediction_2024.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# === 2. Interface calendrier ===
st.sidebar.header("🗓️ Choisir une date")
selected_date = st.sidebar.date_input("Date", df["date"].min())

# === 3. Filtrer les données ===
day_data = df[df["date"] == pd.to_datetime(selected_date)]

def interpret_agri_score(score):
    if score >= 0.75:
        return "🌿 Bonne condition"
    elif score >= 0.4:
        return "🌾 Moyenne condition"
    else:
        return "🌵 Mauvaise condition"

if not day_data.empty:
    row = day_data.iloc[0]

    st.subheader(f"🔎 Résultat pour le {selected_date.strftime('%d %B %Y')}")

    # Traduction des noms d'indicateurs
    indicateur_labels = {
        "soil_m0_7": "Humidité du sol (0-7 cm)",
        "soil_t0_7": "Température du sol (0-7 cm)",
        "agri_score": "Potentiel de production agricole"
    }

    for col in df.columns:
        if col.endswith("_pred"):
            base = col.replace("_pred", "")
            pred_val = row[col]
            real_val = row[f"{base}_real"]
            label = indicateur_labels.get(base, base)

            st.markdown(f"### 📊 {label}")
            st.markdown(f"**Prédit** : {pred_val:.3f}")
            st.markdown(f"**Réel** : {real_val:.3f}")

            # Interprétation spéciale pour agri_score
            if base == "agri_score":
                st.markdown(f"**🧠 Interprétation** : {interpret_agri_score(pred_val)}")
            st.markdown("---")

else:
    st.warning("Aucune donnée disponible pour cette date.")

# === 4. Optionnel : graphique score ===
with st.expander("🔢 Voir toute la courbe des prédictions"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in df.columns:
        if col.endswith("_pred"):
            base = col.replace("_pred", "")
            ax.plot(df["date"], df[col], label=f"{base} (prédit)")
            ax.plot(df["date"], df[f"{base}_real"], linestyle='--', label=f"{base} (réel)")
    ax.set_title("Prédictions vs Valeurs Réelles - Indicateurs du sol et production (2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeurs")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
