# train_model.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback  # type: ignore
from tensorflow.keras import Input  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.set_page_config(page_title="Entraînement du modèle IA", layout="centered")
st.title("⚙️ Entraînement personnalisé du modèle")

# Session state
if "df_result" not in st.session_state:
    st.session_state.df_result = None

st.subheader("Configuration du modèle")
col1, col2 = st.columns(2)

with col1:
    model_name = st.text_input("Nom du modèle (sans extension)", value="mon_modele")
    epochs = st.slider("Nombre d'epochs", min_value=5, max_value=100, value=15, step=5)
    units = st.slider("Unités LSTM", min_value=32, max_value=256, value=150, step=16)
    seq_len = st.slider("Longueur de séquence", min_value=10, max_value=120, value=60, step=10)

with col2:
    batch_size = st.selectbox("Batch size", [16, 25, 32, 64], index=1)
    dropout_rate = st.slider("Taux de Dropout", 0.0, 0.5, 0.1, step=0.05)


class StreamlitLogger(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_log = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_log.text(f"Epoch {epoch+1}/{self.total_epochs} → loss={logs.get('loss'):.4f} - val_loss={logs.get('val_loss'):.4f}")

if st.button("Lancer l'entraînement"):
    progress = st.progress(0)
    step = 0
    total_steps = 6

    st.write("### 1. Chargement des données...")
    train_file = "data/train_with_score.csv"
    test_file = "data/test_2024_with_score.csv"
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        st.error("Les fichiers de données sont manquants.")
        st.stop()
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    step += 1
    progress.progress(step / total_steps)

    st.write("### 2. Normalisation et préparation des données...")
    features = ['precip_mm', 'rain_mm', 'snow_mm', 't2m_max', 't2m_min', 't2m_mean',
                'app_tmax', 'app_tmin', 'sun_h', 'wind10_max', 'gust10_max', 'winddir',
                'sw_rad', 'et0', 'soil_m0_7', 'soil_t0_7']
    targets = ['soil_m0_7', 'soil_t0_7', 'agri_score']
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    X_scaled = scaler_x.fit_transform(full_data[features])
    y_scaled = scaler_y.fit_transform(full_data[targets])
    step += 1
    progress.progress(step / total_steps)

    st.write("### 3. Construction des séquences d'entraînement...")
    X_train, y_train = [], []
    train_len = len(train_data)
    for i in range(seq_len, train_len):
        X_train.append(X_scaled[i-seq_len:i])
        y_train.append(y_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    step += 1
    progress.progress(step / total_steps)

    st.write("### 4. Construction du modèle...")
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(targets)))
    model.compile(optimizer=Adam(0.0003), loss='mse')
    step += 1
    progress.progress(step / total_steps)

    st.write("### 5. Entraînement du modèle...")
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    logger = StreamlitLogger(epochs)
    with st.spinner("Modèle en cours d'entraînement..."):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop, logger],
            verbose=0
        )
    step += 1
    progress.progress(step / total_steps)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.h5"
    model_path = os.path.join("models", model_filename)
    model.save(model_path)
    st.success(f"Modèle sauvegardé sous `{model_path}`")

    st.write("### 6. Prédiction sur les données de test...")
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        seq = scaler_x.transform(test_data[features].iloc[i-seq_len:i])
        X_test.append(seq)
        y_test.append(test_data[targets].iloc[i].values)
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    dates = test_data.iloc[seq_len:].reset_index(drop=True)['date']
    df_result = pd.DataFrame(y_pred, columns=["soil_m0_7_pred", "soil_t0_7_pred", "agri_score_pred"])
    df_result['soil_m0_7_real'] = y_test[:, 0]
    df_result['soil_t0_7_real'] = y_test[:, 1]
    df_result['agri_score_real'] = y_test[:, 2]
    df_result['date'] = pd.to_datetime(dates.values)

    st.session_state.df_result = df_result
    st.success("Prédictions générées sur les données de test")

# === Affichage des résultats ===
if st.session_state.df_result is not None:
    df_result = st.session_state.df_result

    st.write("## 🗓️ Explorer les résultats par date")
    selected_date = st.date_input("Sélectionner une date", df_result['date'].min())
    row = df_result[df_result['date'] == pd.to_datetime(selected_date)]
    if not row.empty:
        r = row.iloc[0]
        st.markdown(f"### Résultats pour le {selected_date.strftime('%d %B %Y')}")
        st.markdown(f"- **Humidité - Prédit** : {r['soil_m0_7_pred']:.3f} / **Réel** : {r['soil_m0_7_real']:.3f}")
        st.markdown(f"- **Temp. sol - Prédit** : {r['soil_t0_7_pred']:.3f} / **Réel** : {r['soil_t0_7_real']:.3f}")
        st.markdown(f"- **Score - Prédit** : {r['agri_score_pred']:.3f} / **Réel** : {r['agri_score_real']:.3f}")

    st.write("## 📈 Comparaison des courbes")
    st.line_chart(df_result.set_index("date")[['soil_m0_7_pred', 'soil_m0_7_real']])
    st.line_chart(df_result.set_index("date")[['soil_t0_7_pred', 'soil_t0_7_real']])
