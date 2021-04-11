import json
import requests
from requests.auth import HTTPBasicAuth

def parser(login, password):
    response = requests.get('http://88.206.16.134:8000/tinvest/market/candles/?figi=BBG004S681W1&from_=2020-05-01T10:05:00&to_=2021-04-10T14:20:00&interval=day', auth=HTTPBasicAuth(login, password))

    with open("data/1.json", "w") as write_file:
        json.dump(response.json(), write_file)
