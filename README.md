#  Prédiction Agricole avec LSTM

Cette application permet de prédire l'humidité et la température du sol à l'aide d'un modèle LSTM, en s'appuyant sur des données météorologiques. Développée avec **Streamlit**, elle propose une interface simple pour entraîner des modèles et générer des prédictions sur une année future.

##  Structure du projet

```
.
├── main.py                # Page d'accueil de l'application
├── train_model.py         # Script Streamlit pour entraîner un modèle
├── predict_future.py      # Script Streamlit pour effectuer une prédiction
├── requirements.txt       # Dépendances Python
├── models/                # Dossier des modèles entraînés (.h5)
├── data/                  # Données d'entraînement/test et prédictions
```

##  Lancer l'application

Installe les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

Lance l'application Streamlit :

```bash
streamlit run main.py
```

Navigue ensuite à travers l'application à l'aide du menu latéral.

##  Fonctionnalités

###  Entraînement personnalisé
- Configuration interactive : epochs, LSTM units, dropout, etc.
- Sauvegarde automatique du modèle entraîné (.h5) dans le dossier `models/`.

###  Prédiction future
- Prédiction quotidienne sur une année complète.
- Comparaison visuelle avec les données réelles (si disponibles).
- Export des résultats au format `.csv`.

##  Données

Les scripts attendent les fichiers suivants dans le dossier `data/` :
- `train_with_score.csv`
- `test_2024_with_score.csv` (et variantes : `test_2025_with_score.csv`, etc.)

Le format attendu contient les colonnes météorologiques (`t2m_max`, `rain_mm`, etc.) ainsi que :
- `soil_m0_7`, `soil_t0_7` (humidité/température du sol)
- `agri_score` (score agronomique global)

##  Modèles

Les modèles sont sauvegardés au format `.h5` dans `models/` et peuvent être rechargés dans l'interface de prédiction.

##  Visualisation

- Prédictions journalières affichées avec des courbes dynamiques.
- Comparaison avec les vraies valeurs si les données existent.

##  Dépendances principales

- `TensorFlow` (modélisation LSTM)
- `scikit-learn` (normalisation)
- `Streamlit` (interface)
- `pandas`, `numpy`, `matplotlib`

---

##  Auteur

Tommy CHOUANGMALA & Baptiste MINET
