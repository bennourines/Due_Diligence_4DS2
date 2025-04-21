import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)

if response.status_code == 200:
    models = response.json()
    # Check if the response is a list or a dictionary
    if isinstance(models, list):
        for model in models:
            print(model)  # Print the model directly if it's a string or dictionary
    elif isinstance(models, dict) and "models" in models:
        for model in models["models"]:  # Adjust based on the actual key
            print(model["id"])
    else:
        print("Unexpected response format:", models)
else:
    print("Error:", response.status_code, response.text)