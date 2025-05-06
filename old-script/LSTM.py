import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("\n=== 1. Chargement des données ===")
train_data = pd.read_csv("train_with_score.csv")
test_data = pd.read_csv("test_2024_with_score.csv")
print("-> Données historiques (multi-années) et test 2024 chargées")

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
print("-> Lignes avec valeurs manquantes supprimées")

print("\n=== 2. Définition des colonnes ===")
feature_columns = [
    # Données météo
    'precip_mm', 'rain_mm', 'snow_mm',
    't2m_max', 't2m_min', 't2m_mean',
    'app_tmax', 'app_tmin', 'sun_h',
    'wind10_max', 'gust10_max', 'winddir',
    'sw_rad', 'et0',
    # Indicateurs sol du passé utilisés comme entrées
    'soil_m0_7', 'soil_t0_7'
]
target_columns = ['soil_m0_7', 'soil_t0_7', 'agri_score']  # Multi-sorties à prédire

print(f"-> Features: {feature_columns}")
print(f"-> Targets: {target_columns}")

print("\n=== 3. Mise à l'échelle ===")
full_data = pd.concat([train_data, test_data], ignore_index=True)
full_data.dropna(subset=target_columns, inplace=True)

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(full_data[feature_columns])
targets_scaled = target_scaler.fit_transform(full_data[target_columns])
print("-> Normalisation effectuée")

print("\n=== 4. Préparation des séquences ===")
sequence_length = 60
train_len = len(train_data)
X_train, y_train = [], []
for i in range(sequence_length, train_len):
    X_train.append(features_scaled[i-sequence_length:i])
    y_train.append(targets_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)
print(f"-> {X_train.shape[0]} séquences générées")

print("\n=== 5. Construction et entraînement du modèle LSTM ===")
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(150))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(target_columns)))  # Multi-sorties
model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=15, batch_size=25, validation_split=0.1, callbacks=[early_stop])
print("-> Entraînement terminé")

print("\n=== 6. Prédiction jour par jour pour 2024 ===")
initial_sequence = features_scaled[train_len - sequence_length:train_len]
current_seq = initial_sequence.copy()
predictions_scaled = []
max_pred_steps = len(features_scaled) - train_len
for i in range(min(365, max_pred_steps)):
    pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
    predictions_scaled.append(pred)
    next_input = current_seq[1:]
    next_day = features_scaled[train_len + i]
    current_seq = np.vstack([next_input, next_day])
print("-> Prédictions complètes")

print("\n=== 7. Inversion de la normalisation des prédictions ===")
predictions = target_scaler.inverse_transform(predictions_scaled)
true_scaled = target_scaler.transform(test_data[target_columns].values[:len(predictions)])
true_2024 = target_scaler.inverse_transform(true_scaled)
dates_2024 = test_data['date'].values[:len(predictions)]
print("-> Valeurs réelles et prédites dénormalisées")

# Pour analyse ou export
result_df = pd.DataFrame(predictions, columns=[f"{col}_pred" for col in target_columns])
for i, col in enumerate(target_columns):
    result_df[f"{col}_real"] = true_2024[:, i]
result_df.insert(0, "date", dates_2024)

result_df.to_csv("soil_multi_prediction_2024.csv", index=False)
print("Résultats sauvegardés dans 'soil_multi_prediction_2024.csv'")

print("\n=== 8. Affichage des courbes ===")
plt.figure(figsize=(12, 6))
for i, col in enumerate(target_columns):
    plt.plot(result_df[f"{col}_pred"], label=f"Prédit - {col}")
    plt.plot(result_df[f"{col}_real"], linestyle='--', label=f"Réel - {col}")
plt.title("Prédictions des indicateurs du sol et score agricole - 2024")
plt.xlabel("Jour")
plt.ylabel("Valeurs physiques / score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
