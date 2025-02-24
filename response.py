import requests
import json

url = "http://127.0.0.1:5000/predict"

payload = {
    "crop_name": "Maize",
    "growth_phase": "Harvesting",
    "temp": 30,
    "humidity": 20,
    "precipitation": 5,
    "moisture": 2,
    "api_temp": [10, 10, 20, 15],
    "api_humidity": [8, 9, 8, 7],
    "api_precipitation": [5, 5, 6, 4]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print(response.json())
