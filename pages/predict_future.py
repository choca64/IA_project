# predict_future.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prédiction future", layout="centered")
st.title("🔮 Prédire l'état du sol pour une année future")

# Initialisation mémoire session
if "df_preds" not in st.session_state:
    st.session_state.df_preds = None
    st.session_state.pred_year = None

# === Configuration utilisateur au centre ===
st.subheader("Configuration de la prédiction")
col1, col2 = st.columns(2)

with col1:
    mode = st.radio("Mode de prédiction", ["Utiliser des données météo", "Simuler sans données"])
    annee = st.number_input("Année à prédire", min_value=2024, max_value=2100, value=2025)

with col2:
    model_files = [f for f in os.listdir("models") if f.endswith(".h5") or f.endswith(".keras")]
    selected_model = st.selectbox("Modèle à utiliser", model_files)

if st.button("Lancer la prédiction"):
    st.write(f"### 🔧 Modèle utilisé : `{selected_model}`")
    st.write(f"### 🕒 Année à prédire : `{annee}`")

    with st.spinner("Chargement du modèle et des scalers..."):
        try:
            model = load_model(f"models/{selected_model}", compile=False)
            st.write("✅ Modèle chargé avec succès")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            st.stop()

        train_df = pd.read_csv("data/train_with_score.csv")
        test_file = f"data/test_{annee - 1}_with_score.csv"

        if not os.path.exists(test_file):
            st.error(f"Le fichier {test_file} est introuvable. Veuillez le générer.")
            st.stop()

        test_df = pd.read_csv(test_file)
        st.write("✅ Données d'entraînement et test chargées")

        scaler_x = MinMaxScaler().fit(train_df[[
            'precip_mm', 'rain_mm', 'snow_mm', 't2m_max', 't2m_min', 't2m_mean',
            'app_tmax', 'app_tmin', 'sun_h', 'wind10_max', 'gust10_max', 'winddir',
            'sw_rad', 'et0', 'soil_m0_7', 'soil_t0_7']])
        scaler_y = MinMaxScaler().fit(train_df[['soil_m0_7', 'soil_t0_7', 'agri_score']])
        st.write("✅ Normalisation des données effectuée")

    st.write(f"### 🔄 Génération des données météo simulées pour {annee}")
    last_vals = test_df.iloc[-1]
    dates = [datetime(annee, 1, 1) + timedelta(days=i) for i in range(365)]

    df_future = pd.DataFrame({"date": dates})
    for col in [
        'precip_mm', 'rain_mm', 'snow_mm', 't2m_max', 't2m_min', 't2m_mean',
        'app_tmax', 'app_tmin', 'sun_h', 'wind10_max', 'gust10_max', 'winddir',
        'sw_rad', 'et0', 'soil_m0_7', 'soil_t0_7']:
        df_future[col] = last_vals[col] + np.random.normal(0, 0.1, len(df_future))

    st.write("✅ Données météo simulées avec variations aléatoires")

    st.write("### 🤖 Prédiction IA jour par jour avec feedback")
    full_input = pd.concat([train_df, test_df, df_future], ignore_index=True)
    full_scaled = scaler_x.transform(full_input[[
        'precip_mm', 'rain_mm', 'snow_mm', 't2m_max', 't2m_min', 't2m_mean',
        'app_tmax', 'app_tmin', 'sun_h', 'wind10_max', 'gust10_max', 'winddir',
        'sw_rad', 'et0', 'soil_m0_7', 'soil_t0_7']])

    seq_len = 60
    predictions_scaled = []
    start_idx = len(full_scaled) - len(df_future) - seq_len
    current_seq = full_scaled[start_idx:start_idx + seq_len]

    for i in range(len(df_future)):
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
        predictions_scaled.append(pred)
        st.write(f"📅 {df_future['date'].iloc[i].strftime('%d/%m/%Y')} → 🌱 Humidité : {pred[0]:.3f}, 🌡️ Temp sol : {pred[1]:.3f}, 🧮 Score : {pred[2]:.3f}")

        next_input = full_scaled[start_idx + seq_len + i].copy()
        next_input[-2:] = pred[:2]  # feedback sur humidité/temp
        current_seq = np.vstack([current_seq[1:], next_input])

    preds = scaler_y.inverse_transform(predictions_scaled)
    df_preds = pd.DataFrame(preds, columns=['soil_m0_7_pred', 'soil_t0_7_pred', 'agri_score_pred'])
    df_preds['date'] = pd.to_datetime(df_future['date'])

    st.session_state.df_preds = df_preds
    st.session_state.pred_year = annee
    pred_file = f"data/prediction_{annee}.csv"
    df_preds.to_csv(pred_file, index=False)
    st.success(f"✅ Prédiction enregistrée dans `{pred_file}`")

# === Affichage ===
if st.session_state.df_preds is not None:
    df_preds = st.session_state.df_preds

    st.write("## 🗓️ Explorer les prédictions par date")
    selected_date = st.date_input("Sélectionner une date", df_preds['date'].min())
    row = df_preds[df_preds['date'] == pd.to_datetime(selected_date)]

    if not row.empty:
        row = row.iloc[0]
        st.markdown(f"### Résultats pour le {selected_date.strftime('%d %B %Y')}")
        st.markdown(f"- **Humidité du sol** : {row['soil_m0_7_pred']:.3f}")
        st.markdown(f"- **Température du sol** : {row['soil_t0_7_pred']:.3f}")
        st.markdown(f"- **Score agricole estimé** : {row['agri_score_pred']:.3f}")
    else:
        st.warning("Aucune donnée pour cette date.")

    st.write("## 📈 Visualisation annuelle")
    st.line_chart(df_preds.set_index("date"))
