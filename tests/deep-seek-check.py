import os
import requests
from pprint import pprint

# api_key = os.getenv("DEEPSEEK_API_KEY")
api_key = "sk-32840580d42b43f98dfd4d6a3fe9f9c7"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print(headers)

response = requests.post("https://api.deepseek.com/v1/endpoint", headers=headers, json={"query": "test"})
print(response.json())
# pprint(response)
