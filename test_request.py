import requests

# Your FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# sample --> 8890000,4600,3,2,2,2,yes,yes,no,no,yes,no,furnished

# Sample input data (replace these with real test values)
data = {
    "area": 4600,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 2,
    "mainroad": 'yes',
    "guestroom": 'yes',
    "basement": 'no',
    "hotwaterheating": 'no',
    "airconditioning": 'yes',
    "prefarea": 'no',
    "furnishingstatus": 'furnished'
}

# Send POST request
response = requests.post(url, json=data)

# Show result
print("Status Code:", response.status_code)
print("Response JSON:", (int((response.json()['predicted_price'] * 11000000) + 175000)), '$')