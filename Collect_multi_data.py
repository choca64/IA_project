import os
import requests
import pandas as pd
from datetime import datetime

# === 1. Altitude simulée ===
def fetch_elevation(lat: float, lon: float) -> float:
    print("→ Altitude simulée utilisée (remplacer plus tard par une vraie API)")
    return 135.0

# === 2. Données météo via Open-Meteo ===
def fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    daily_vars = ",".join([
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min",
        "precipitation_sum", "rain_sum", "snowfall_sum",
        "sunshine_duration", "windspeed_10m_max", "windgusts_10m_max",
        "winddirection_10m_dominant", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "soil_temperature_0_to_7cm_mean",
        "soil_moisture_0_to_7cm_mean",
    ])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": daily_vars,
        "timezone": "Europe/Paris",
    }

    print("→ Téléchargement des données Open-Meteo…")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame(data).rename(columns={
        "time": "date", "temperature_2m_max": "t2m_max", "temperature_2m_min": "t2m_min",
        "temperature_2m_mean": "t2m_mean", "apparent_temperature_max": "app_tmax",
        "apparent_temperature_min": "app_tmin", "precipitation_sum": "precip_mm",
        "rain_sum": "rain_mm", "snowfall_sum": "snow_mm", "sunshine_duration": "sun_h",
        "windspeed_10m_max": "wind10_max", "windgusts_10m_max": "gust10_max",
        "winddirection_10m_dominant": "winddir", "shortwave_radiation_sum": "sw_rad",
        "et0_fao_evapotranspiration": "et0", "soil_temperature_0_to_7cm_mean": "soil_t0_7",
        "soil_moisture_0_to_7cm_mean": "soil_m0_7",
    })

    df["date"] = pd.to_datetime(df["date"])
    print(f"   ✅ {len(df)} lignes récupérées")
    return df

# === 3. NDVI simulé ===
def fetch_mock_ndvi(lat: float, lon: float, date: str) -> float:
    return 0.65  # Valeur fixe pour démo

# === 4. API OpenLandMap pour sol (pH, MO, texture) ===
def get_soil_data(lat: float, lon: float) -> dict:
    layers = {
        "ph_0_5cm": "phh2o_mean_0-5cm",
        "ocd_0_5cm": "ocd_mean_0-5cm",
        "clay_0_5cm": "clay_mean_0-5cm",
        "sand_0_5cm": "sand_mean_0-5cm",
    }

    soil_data = {}
    print("→ Requête API OpenLandMap (sol) en cours…")

    for label, layer in layers.items():
        url = f"https://landgisapi.opengeohub.org/query?lon={lon}&lat={lat}&layer={layer}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                val = r.json().get("value")
                soil_data[label] = val
            else:
                soil_data[label] = None
        except Exception as e:
            print(f"   ⚠️ Erreur {label}: {e}")
            soil_data[label] = None

    return soil_data

# === 5. Pipeline principal ===
def run_pipeline(lat: float, lon: float, start: str, end: str, output_file: str) -> pd.DataFrame:
    print("=== Début du pipeline ===")
    
    elevation = fetch_elevation(lat, lon)
    df_weather = fetch_open_meteo(lat, lon, start, end)
    soil = get_soil_data(lat, lon)

    print("→ Ajout des colonnes Latitude, NDVI, Altitude, et données du sol…")
    df_weather["latitude"] = lat
    df_weather["longitude"] = lon
    df_weather["elevation_m"] = elevation
    df_weather["ndvi"] = [fetch_mock_ndvi(lat, lon, d.strftime("%Y-%m-%d")) for d in df_weather["date"]]

    for key, value in soil.items():
        df_weather[key] = value

    df_weather.to_csv(output_file, index=False)
    print(f"✅ {len(df_weather)} lignes exportées dans le fichier.")
    print(f"✅ CSV généré : {output_file} ({os.path.getsize(output_file)} octets)")
    print("=== Pipeline terminé ===")
    return df_weather

# === EXÉCUTION ===
if __name__ == "__main__":
    LAT, LON = 43.6045, 1.4440  # Toulouse
    START, END = "2023-01-01", "2023-12-31"
    OUT_CSV = "merge_data.csv"

    run_pipeline(LAT, LON, START, END, OUT_CSV)
