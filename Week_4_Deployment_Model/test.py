import requests
import sys

ride = {
    "year": int(sys.argv[1]),
    "month": int(sys.argv[2]),
    "model" : "model.bin"
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
