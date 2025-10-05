import requests

url = "http://127.0.0.1:8000/predict"

sample = {"features" : {"bedroom" : 3, "bathroom" : 2.0, "area" : 300}}

response = requests.post(url, json=sample)

print("status_code:", response.status_code)
print("Response json:", response.json())