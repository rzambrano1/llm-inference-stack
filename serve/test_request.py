import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Once upon a time there was a dog that could pilot helicopters",
        "max_new_tokens": 100
    }
)

print(response.json())