import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class OpenRouterChatbot:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:5000",  # Your site URL
            "Content-Type": "application/json"
        }

    def chat(self, message: str, model: str = "anthropic/claude-2", system_prompt: str = None) -> Dict[Any, Any]:
        """
        Send a chat message to the specified model via OpenRouter
        
        Args:
            message (str): User's message
            model (str): Model identifier (default: anthropic/claude-2)
            system_prompt (str): Optional system prompt to guide the model's behavior
        
        Returns:
            Dict: Response from the model
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        
        if system_prompt:
            payload["messages"].insert(0, {
                "role": "system",
                "content": system_prompt
            })

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

def main():
    # Initialize chatbot
    chatbot = OpenRouterChatbot()
    
    print("OpenRouter Chatbot (type 'quit' to exit)")
    print("Available models:")
    print("- nvidia/llama-3.3-nemotron-super-49b-v1:free (default)")
    print("- anthropic/claude-2")
    print("- google/palm-2-chat-bison")
    print("- openai/gpt-3.5-turbo")
    print("- openai/gpt-4")
    
    # Get model choice
    model = input("\nChoose a model (press Enter for default nemotron): ").strip()
    if not model:
        model = "nvidia/llama-3.3-nemotron-super-49b-v1:free"
    
    # Optional system prompt
    system_prompt = input("Enter system prompt (optional, press Enter to skip): ").strip()
    if not system_prompt:
        system_prompt = None
    
    # Main chat loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
            
        try:
            response = chatbot.chat(user_input, model, system_prompt)
            assistant_message = response['choices'][0]['message']['content']
            print(f"\nAssistant: {assistant_message}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()