import requests
import pandas as pd
from datetime import datetime
import os

CITIES = {
    "Prishtine": {"lat": 42.6629, "lon": 21.1655},
    "Peje": {"lat": 42.6591, "lon": 20.2883},
    "Prizren": {"lat": 42.2139, "lon": 20.7397}
}

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

os.makedirs("../Datasetet/unprocessed_datasets", exist_ok=True)

for city, coords in CITIES.items():
    print(f"\nProcessing: {city}")

    air_file = f"../Datasetet/unprocessed_datasets/{city.lower()}_air_quality_sat.csv"
    weather_file = f"../Datasetet/unprocessed_datasets/{city.lower()}_weather_raw.csv"

    print("Fetching Air Quality data...")
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ["pm10", "pm2_5"],
        "timezone": "auto"
    }

    try:
        a_resp = requests.get(air_url, params=air_params)
        a_data = a_resp.json()

        if "hourly" in a_data:
            df_air = pd.DataFrame(a_data["hourly"])
            df_air.rename(columns={"time": "date"}, inplace=True)
            df_air.to_csv(air_file, index=False)
            print(f"Saved {len(df_air)} air records to {air_file}")
        else:
            print(f"Air error for {city}: {a_data}")
    except Exception as e:
        print(f"Critical air error for {city}: {e}")

    print("Fetching Weather data...")
    weather_url = "https://archive-api.open-meteo.com/v1/archi.ve"
    weather_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"],
        "timezone": "auto"
    }

    try:
        w_resp = requests.get(weather_url, params=weather_params)
        w_data = w_resp.json()

        if "hourly" in w_data:
            df_weather = pd.DataFrame(w_data["hourly"])
            df_weather.rename(columns={"time": "date"}, inplace=True)
            df_weather.to_csv(weather_file, index=False)
            print(f"Saved {len(df_weather)} weather records to {weather_file}")
        else:
            print(f"Weather error for {city}: {w_data}")
    except Exception as e:
        print(f"Critical weather error for {city}: {e}")

print("\nAll tasks completed successfully!")