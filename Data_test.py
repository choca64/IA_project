import requests
import pandas as pd

# === 1. Fonction Open-Meteo (2024) ===
def fetch_open_meteo_2024(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min",
            "precipitation_sum", "rain_sum", "snowfall_sum",
            "sunshine_duration", "windspeed_10m_max", "windgusts_10m_max",
            "winddirection_10m_dominant", "shortwave_radiation_sum",
            "et0_fao_evapotranspiration", "soil_temperature_0_to_7cm_mean",
            "soil_moisture_0_to_7cm_mean",
        ]),
        "timezone": "Europe/Paris",
    }

    print("📱 Téléchargement des données Open-Meteo 2024…")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame(data).rename(columns={
        "time": "date",
        "temperature_2m_max": "t2m_max",
        "temperature_2m_min": "t2m_min",
        "temperature_2m_mean": "t2m_mean",
        "apparent_temperature_max": "app_tmax",
        "apparent_temperature_min": "app_tmin",
        "precipitation_sum": "precip_mm",
        "rain_sum": "rain_mm",
        "snowfall_sum": "snow_mm",
        "sunshine_duration": "sun_h",
        "windspeed_10m_max": "wind10_max",
        "windgusts_10m_max": "gust10_max",
        "winddirection_10m_dominant": "winddir",
        "shortwave_radiation_sum": "sw_rad",
        "et0_fao_evapotranspiration": "et0",
        "soil_temperature_0_to_7cm_mean": "soil_t0_7",
        "soil_moisture_0_to_7cm_mean": "soil_m0_7",
    })

    df["date"] = pd.to_datetime(df["date"])
    return df

# === 2. Fonction score de production agricole ===
def compute_agri_score(row):
    score = 1.0

    if row['precip_mm'] < 1:
        score -= 0.4
    elif row['precip_mm'] > 10:
        score += 0.2

    if 15 <= row['t2m_mean'] <= 30:
        score += 0.2
    else:
        score -= 0.3

    if 0.15 <= row['soil_m0_7'] <= 0.35:
        score += 0.3
    else:
        score -= 0.3

    if row['sun_h'] > 10:
        score += 0.1

    return max(0, min(score, 1))

# === 3. Exécution ===
if __name__ == "__main__":
    LAT, LON = 43.6045, 1.4440  # Toulouse
    print("=== Génération des données météo + sol (2024) avec score de production agricole ===")
    df = fetch_open_meteo_2024(LAT, LON)
    df["agri_score"] = df.apply(compute_agri_score, axis=1)
    df.to_csv("test_2024_with_score.csv", index=False)
    print("✅ Fichier généré : test_2024_with_score.csv avec colonne agri_score")
