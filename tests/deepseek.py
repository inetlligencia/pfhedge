import requests
import json

API_KEY = "sk-32840580d42b43f98dfd4d6a3fe9f9c7"
# "sk-64788cf20f224b3abbbf58c4cea2cb6f"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

body = {
    "messages": [
        {
            "role": "system",
            "content": "You are a test assistant."
        },
        {
            "role": "user", 
            "content": "Testing. Just say hi and nothing else."
        }
    ],
    "model": "deepseek-chat",
    # "temperature": 0.7,
    # "max_tokens": 100
}

response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers=headers,
    json=body
)

# Print both status code and response
print(f"Status code: {response.status_code}")
try:
    print("Response:", response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Error decoding JSON response:", str(e))
    print("Raw response:", response.text)
