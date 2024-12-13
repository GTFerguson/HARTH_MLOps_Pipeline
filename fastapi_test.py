import requests

url = "http://localhost:8000/predict"
payload = {"features": [0.5, 1.2, 3.4, 0.5, 3.4, 2.9]}  # Replace with valid data structure
response = requests.post(url, json=payload)

# Inspect the status code and raw response content
print("Status Code:", response.status_code)
print("Response Content:", response.text)

# If the response is valid, decode it
try:
    response_data = response.json()
    print("Response JSON:", response_data)
except requests.exceptions.JSONDecodeError as e:
    print("JSON Decode Error:", e)