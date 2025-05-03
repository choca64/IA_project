import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

print("\n=== 1. Chargement des données ===")
train_data = pd.read_csv("all_years_data_train.csv", sep=',')
test_data = pd.read_csv("2024_data_test.csv", sep=',')
print("-> Données historiques (multi-années) et test 2024 chargées")

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
print("-> Lignes avec valeurs manquantes supprimées")


print("\n=== 2. Définition des colonnes ===")
feature_columns = [
    'precip_mm', 'rain_mm', 'snow_mm',
    't2m_max', 't2m_min', 't2m_mean',
    'app_tmax', 'app_tmin',
    'sun_h', 'wind10_max', 'gust10_max', 'winddir',
    'sw_rad', 'et0'
]
target_columns = ['soil_m0_7']  # Prédiction de l'humidité du sol
print(f"-> Features: {feature_columns}")
print(f"-> Target: {target_columns}")
print("\n=== PARAMÈTRES D'ENTRAÎNEMENT ===")
print(f"Nombre total de features utilisées : {len(feature_columns)}")
print("Colonnes utilisées pour l'entraînement (features) :")
for col in feature_columns:
    print(f" - {col}")
print(f"\nTarget à prédire : {target_columns[0]}")

print("\n=== 3. Mise à l'échelle ===")
full_data = pd.concat([train_data, test_data], ignore_index=True)

# Supprimer les lignes avec NaN dans les colonnes cibles
full_data.dropna(subset=target_columns, inplace=True)

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(full_data[feature_columns])
targets_scaled = target_scaler.fit_transform(full_data[target_columns])
print("-> Normalisation des features et targets effectuée")

# DEBUG : vérification des targets
print("Cibles (targets) échantillon :", targets_scaled[:5])
print("Min / Max soil_m0_7 :", full_data['soil_m0_7'].min(), "/", full_data['soil_m0_7'].max())
print("Valeurs uniques soil_m0_7 :", full_data['soil_m0_7'].nunique())

print("\n=== 4. Préparation des séquences ===")
sequence_length = 60  # Nombre de jours pour prédire le jour suivant
train_len = len(train_data) 
X_train, y_train = [], []
for i in range(sequence_length, train_len):
    X_train.append(features_scaled[i-sequence_length:i])
    y_train.append(targets_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)
print(f"-> {X_train.shape[0]} séquences d'entraînement générées")

print("\n=== 5. Construction et entraînement du modèle LSTM ===")
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.05))  # ➜ 5% des neurones sont désactivés aléatoirement
model.add(LSTM(150))
model.add(Dense(32, activation='relu'))  # Ajoute avant le Dense(1)
model.add(Dense(1))  # Une seule sortie : humidité du sol
model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')  # ➜ learning rate ajusté

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
print("-> Prédictions complètes pour les jours disponibles")

print("\n=== 7. Inversion de la normalisation des prédictions ===")
predictions = target_scaler.inverse_transform(predictions_scaled)
true_2024 = test_data['soil_m0_7'].values[:len(predictions)]
print("-> Données prédites et réelles prêtes pour comparaison")

print("\n=== DEBUG VISUEL ===")
print("Shape prédictions :", predictions.shape)
print("Extrait prédictions :", predictions[:5])
print("Shape vrai 2024 :", true_2024.shape)
print("Extrait vrai 2024 :", true_2024[:5])

print("\n=== 8. Affichage des courbes de comparaison ===")
plt.figure(figsize=(12, 5))
plt.plot(predictions[:, 0], label='Prédit - Humidité sol (0-7 cm)')
plt.plot(true_2024, label='Réel - Humidité sol (0-7 cm)')
plt.legend()
plt.title("Prédiction de l'Humidité du Sol (2024)")
plt.show()
print("-> Affichage terminé ✅")
