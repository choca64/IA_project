import os
import requests
import pandas as pd

def fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    daily_vars = ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "sunshine_duration",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "winddirection_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "soil_temperature_0_to_7cm_mean",
        "soil_moisture_0_to_7cm_mean",
    ])

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      daily_vars,
        "timezone":   "Europe/Paris",
    }

    print("→ Téléchargement des données Open-Meteo…")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame(data).rename(columns={
        "time":                          "date",
        "temperature_2m_max":            "t2m_max",
        "temperature_2m_min":            "t2m_min",
        "temperature_2m_mean":           "t2m_mean",
        "apparent_temperature_max":      "app_tmax",
        "apparent_temperature_min":      "app_tmin",
        "precipitation_sum":             "precip_mm",
        "rain_sum":                      "rain_mm",
        "snowfall_sum":                  "snow_mm",
        "sunshine_duration":             "sun_h",
        "windspeed_10m_max":             "wind10_max",
        "windgusts_10m_max":             "gust10_max",
        "winddirection_10m_dominant":    "winddir",
        "shortwave_radiation_sum":       "sw_rad",
        "et0_fao_evapotranspiration":    "et0",
        "soil_temperature_0_to_7cm_mean":"soil_t0_7",
        "soil_moisture_0_to_7cm_mean":   "soil_m0_7",
    })

    df["date"] = pd.to_datetime(df["date"])
    print(f"   {len(df)} lignes récupérées")
    return df

if __name__ == "__main__":
    # 🚩 À adapter selon votre point d'étude
    LAT, LON   = 43.6045, 1.4440
    START, END = "2023-01-01", "2023-12-31"
    OUT_CSV    = "2023_data_train.csv"

    # 1) Récupération Open-Meteo
    df = fetch_open_meteo(LAT, LON, START, END)

    # 2) Export CSV
    df.to_csv(OUT_CSV, index=False, sep=",")
    size = os.path.getsize(OUT_CSV)
    print(f"✅ CSV généré : {OUT_CSV} ({size} octets)")
