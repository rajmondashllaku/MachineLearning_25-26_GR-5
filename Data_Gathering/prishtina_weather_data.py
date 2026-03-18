import requests
import pandas as pd
from datetime import datetime

CITY = "Pristina"

LAT = 42.6629
LON = 21.1655

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

AIR_FILE = "prishtina_air_quality_raw_2020_now.csv"
WEATHER_FILE = "../Datasetet/unprocessed_datasets/prishtina_weather_raw_2020_now.csv"
print("Fetching air quality data...")

url = "https://api.openaq.org/v2/measurements"

limit = 10000
page = 1

all_records = []

while True:

    params = {
        "city": CITY,
        "date_from": START_DATE,
        "date_to": END_DATE,
        "limit": limit,
        "page": page,
        "parameter": ["pm25", "pm10", "no2", "so2", "co"]
    }

    response = requests.get(url, params=params)

    data = response.json()

    results = data.get("results", [])

    if len(results) == 0:
        break

    all_records.extend(results)

    print(f"Fetched page {page} ({len(results)} records)")

    page += 1


print(f"Total air quality records: {len(all_records)}")

df_air = pd.json_normalize(all_records)

df_air.to_csv(AIR_FILE, index=False)

print(f"Saved: {AIR_FILE}")

print("\nFetching weather data...")

weather_url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m"
    ],
    "timezone": "auto"
}

response = requests.get(weather_url, params=params)

weather_data = response.json()

df_weather = pd.DataFrame(weather_data["hourly"])

print(f"Weather rows: {len(df_weather)}")

df_weather.to_csv(WEATHER_FILE, index=False)

print(f"Saved: {WEATHER_FILE}")

print("\nData collection complete.")

print("Files created:")
print(AIR_FILE)
print(WEATHER_FILE)