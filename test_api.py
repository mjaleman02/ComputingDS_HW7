import requests
import json

# URL of the API
url = "http://localhost:8000/predict"

# Load example input data
with open('example_input.json', 'r') as f:
    input_data = json.load(f)

# Send POST request
response = requests.post(url, json=input_data)

# Print the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
