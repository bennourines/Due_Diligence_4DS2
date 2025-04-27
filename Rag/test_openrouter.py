# filepath: c:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\Rag\test_openrouter.py
import requests
import os
from dotenv import load_dotenv

# Load environment variables (ensure .env is in the Rag directory or parent)
load_dotenv(override=True) 

# --- Configuration ---
# Use the same key and model as in RagAndMetrics.py
API_KEY = os.getenv("OPENROUTER_API_KEY", "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") 
LLM_MODEL = "deepseek/deepseek-r1:free" # The model causing issues
TEST_PROMPT = "Explain the concept of Proof-of-Work in simple terms."
# ---

if not API_KEY or "YOUR_API_KEY" in API_KEY:
    print("ERROR: OpenRouter API Key not found or not set. Please check environment variables or .env file.")
else:
    print(f"Using Model: {LLM_MODEL}")
    # Mask most of the key for printing
    masked_key = API_KEY[:10] + "..." + API_KEY[-4:]
    print(f"Using API Key: {masked_key}")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 150,
        "temperature": 0.5
    }

    print("\n--- Sending Request ---")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Data: {data}")
    print("-----------------------\n")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60) # Added timeout

        print(f"--- Response ---")
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"\nBody Text:\n{response.text}")
        print("----------------\n")

        # Try to parse JSON
        try:
            response_json = response.json()
            print("JSON Parsed Successfully:")
            import json
            print(json.dumps(response_json, indent=2))

            # Check for 'choices' specifically
            if "choices" not in response_json:
                print("\nERROR: 'choices' key is missing in the JSON response!")
            elif not response_json["choices"]:
                 print("\nERROR: 'choices' key is present but empty in the JSON response!")
            else:
                 print("\nSUCCESS: 'choices' key found.")
                 # print(f"Generated Text: {response_json['choices'][0].get('message', {}).get('content', 'N/A')}")


        except requests.exceptions.JSONDecodeError:
            print("\nERROR: Failed to decode response body as JSON.")

    except requests.exceptions.RequestException as req_err:
        print(f"\n--- REQUEST FAILED ---")
        print(f"Error Type: {type(req_err)}")
        print(f"Error: {req_err}")
        if hasattr(req_err, 'response') and req_err.response is not None:
            print(f"Response Status Code: {req_err.response.status_code}")
            print(f"Response Text: {req_err.response.text}")
        print("---------------------\n")
    except Exception as e:
         print(f"\n--- UNEXPECTED ERROR ---")
         print(f"Error Type: {type(e)}")
         print(f"Error: {e}")
         print("-----------------------\n")


