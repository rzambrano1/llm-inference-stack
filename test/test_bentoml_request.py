import requests

response = requests.post(
    "http://localhost:3000/generate",
    json={
        "request": {
            "prompt": "Once upon a time there was a dog that could pilot helicopters",
        "max_new_tokens": 100
        }
    }
)

print(response.json())