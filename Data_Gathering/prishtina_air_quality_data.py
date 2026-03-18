import requests
import pandas as pd
from tqdm import tqdm

API_KEY = "d01ce4eba97f24d77eb2054320a4837de981c730b6a52c79450e6241ecbd034a"
headers = {"X-API-Key": API_KEY}
OUTPUT_CSV = "prishtina_air_quality.csv"
DATE_FROM = "2020-01-01T00:00:00Z"
PAGE_LIMIT = 1000

loc_resp = requests.get(
    "https://api.openaq.org/v3/locations",
    headers=headers,
    params={"country": "XK", "city": "Prishtina", "limit": 100}
)

locations = loc_resp.json().get("results", [])
if not locations:
    print("No locations found for Prishtina")
    exit()

sensor_ids = []
for loc in locations:
    for s in loc.get("sensors", []):
        pname = s["parameter"]["name"]
        if pname in ["pm25", "pm10"]:
            sensor_ids.append(s["id"])

print("Found sensor IDs:", sensor_ids)

all_data = []

for sid in tqdm(sensor_ids):
    page = 1
    while True:
        url = f"https://api.openaq.org/v3/sensors/{sid}/measurements"
        params = {
            "limit": PAGE_LIMIT,
            "page": page,
            "datetime_from": DATE_FROM,
        }
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            print(f"Failed at sensor {sid} page {page}")
            break

        results = r.json().get("results", [])
        if not results:
            break

        all_data.extend(results)
        page += 1

print("Total rows:", len(all_data))

if all_data:
    df = pd.json_normalize(all_data)
    df["date.utc"] = pd.to_datetime(df["date.utc"], errors="coerce")
    df = df.sort_values("date.utc")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved to", OUTPUT_CSV)
else:
    print("No measurement data found.")