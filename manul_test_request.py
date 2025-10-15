import requests

# Your FastAPI endpoint
url = "http://127.0.0.1:8080/predict"

# sample --> 8890000,4600,3,2,2,2,yes,yes,no,no,yes,no,furnished
# sample2 --> 10850000,7500,3,3,1,yes,no,yes,no,yes,2,yes,semi-furnished

# Sample input data (replace these with real test values)
data = {
    "area": 7500,
    "bedrooms": 3,
    "bathrooms": 3,
    "stories": 1,
    "parking": 2,
    "mainroad": 'yes',
    "guestroom": 'no',
    "basement": 'yes',
    "hotwaterheating": 'no',
    "airconditioning": 'yes',
    "prefarea": 'yes',
    "furnishingstatus": 'semi-furnished'
}

# Send POST request
response = requests.post(url, json=data)

# Show result
print("Status Code:", response.status_code)
print("Response JSON:", response.json())