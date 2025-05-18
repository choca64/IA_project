# predict_future.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prédiction future", layout="centered")
st.title("🔮 Prédire l'humidité et la température du sol pour une année future")

if "df_preds" not in st.session_state:
    st.session_state.df_preds = None
    st.session_state.pred_year = None

st.subheader("Configuration de la prédiction")
col1, col2 = st.columns(2)

with col1:
    annee = st.number_input("Année à prédire", min_value=2024, max_value=2100, value=2025)

with col2:
    model_files = [f for f in os.listdir("models") if f.endswith(".h5") or f.endswith(".keras")]
    selected_model = st.selectbox("Modèle à utiliser", model_files)

if st.button("Lancer la prédiction"):
    st.write(f"### 🔧 Modèle utilisé : `{selected_model}`")
    st.write(f"### 🕒 Année à prédire : `{annee}`")

    with st.spinner("Chargement du modèle et des données..."):
        try:
            model = load_model(f"models/{selected_model}", compile=False)
            with st.expander("🧠 Résumé du modèle"):
                model.summary(print_fn=st.text)
            st.write("✅ Modèle chargé avec succès")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            st.stop()

        try:
            train_df = pd.read_csv("data/train_with_score.csv")
            test_file = f"data/test_{annee - 1}_with_score.csv"
            test_df = pd.read_csv(test_file)
            st.write(f"✅ Données chargées : train={len(train_df)}, test={len(test_df)}")
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            st.stop()

        features = [
            'precip_mm', 'rain_mm', 'snow_mm', 't2m_max', 't2m_min', 't2m_mean',
            'app_tmax', 'app_tmin', 'sun_h', 'wind10_max', 'gust10_max', 'winddir',
            'sw_rad', 'et0', 'soil_m0_7', 'soil_t0_7']
        targets = ['soil_m0_7', 'soil_t0_7', 'agri_score']

        # Vérification de cohérence des colonnes
        for col in features + targets:
            if col not in train_df.columns or col not in test_df.columns:
                st.error(f"Colonne manquante dans les données : {col}")
                st.stop()

        scaler_x = MinMaxScaler().fit(train_df[features])
        scaler_y = MinMaxScaler().fit(train_df[targets])
        st.write("✅ Normalisation effectuée")

    st.write("### 🔄 Préparation des données de prédiction")
    future_dates = [datetime(annee, 1, 1) + timedelta(days=i) for i in range(365)]

    df_future = test_df.copy()
    for i in range(len(future_dates)):
        df_future = pd.concat([df_future, pd.DataFrame([df_future.iloc[-1]])], ignore_index=True)
        df_future.at[df_future.index[-1], 'date'] = future_dates[i].strftime("%Y-%m-%d")

    full_input = pd.concat([train_df, test_df, df_future], ignore_index=True)
    full_scaled = scaler_x.transform(full_input[features])
    st.write(f"✅ Forme des données d'entrée : {full_scaled.shape}")

    st.write("### 🤖 Prédiction en cours...")
    seq_len = 60
    predictions_scaled = []
    start_idx = len(full_scaled) - len(df_future) - seq_len
    current_seq = full_scaled[start_idx:start_idx + seq_len]

    if current_seq.shape != (seq_len, len(features)):
        st.error(f"Erreur de forme : attendu ({seq_len}, {len(features)}), obtenu {current_seq.shape}")
        st.stop()

    progress_bar = st.progress(0)

    for i in range(len(future_dates)):
        try:
            input_seq = current_seq[np.newaxis, :, :]  # (1, 60, n_features)
            pred = model.predict(input_seq, verbose=0)[0]
            predictions_scaled.append(pred)

            next_input = full_scaled[start_idx + seq_len + i].copy()
            next_input[-2:] = pred[:2]  # mise à jour soil_m0_7 et soil_t0_7
            current_seq = np.vstack([current_seq[1:], next_input])
            progress_bar.progress((i + 1) / len(future_dates))
        except Exception as e:
            st.error(f"Erreur pendant la prédiction à l'itération {i} : {e}")
            st.stop()

    preds = scaler_y.inverse_transform(predictions_scaled)
    df_preds = pd.DataFrame(preds, columns=['soil_m0_7_pred', 'soil_t0_7_pred', 'agri_score_pred'])
    df_preds['date'] = pd.to_datetime(future_dates)

    st.session_state.df_preds = df_preds
    st.session_state.pred_year = annee

    output_path = f"data/prediction_{annee}.csv"
    df_preds.to_csv(output_path, index=False)
    st.success(f"✅ Prédictions sauvegardées dans `{output_path}`")

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
    st.line_chart(df_preds.set_index("date")['soil_m0_7_pred'])
    st.line_chart(df_preds.set_index("date")['soil_t0_7_pred'])

    compare_file = f"data/test_{annee}_with_score.csv"
    if os.path.exists(compare_file):
        real_df = pd.read_csv(compare_file)
        real_df['date'] = pd.to_datetime(real_df['date'])
        merged = pd.merge(df_preds, real_df, on="date", suffixes=("_pred", "_real"))

        st.write("## 🔍 Comparaison avec les données réelles")
        st.line_chart(merged.set_index("date")[['soil_m0_7_pred', 'soil_m0_7']])
        st.line_chart(merged.set_index("date")[['soil_t0_7_pred', 'soil_t0_7']])
