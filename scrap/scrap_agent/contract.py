# contract.py

from fadhli_proxy import Proxy
import requests
from bs4 import BeautifulSoup
import json

# Initialize the proxy
proxy = Proxy(autoRotate=True).proxy()
proxy_url = f"{proxy[1]}://{proxy[0]}"  # Format: http://ip:port

# Define the stock market website URL
stock_url = "https://coinmarketcap.com/"  # Replace with the actual URL

# Define headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Function to scrape the stock market website
def scrape_stock_market():
    try:
        # Make a request to the stock market website using the proxy
        response = requests.get(stock_url, headers=headers, proxies={"http": proxy_url, "https": proxy_url}, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract relevant data (adjust based on the website's structure)
            stock_data = []
            for row in soup.select("table.stocks tr"):  # Adjust the selector based on the website's HTML
                cells = row.find_all("td")
                if len(cells) > 1:
                    stock_data.append({
                        "symbol": cells[0].text.strip(),
                        "price": cells[1].text.strip(),
                        "change": cells[2].text.strip()
                    })
            
            # Save the scraped data to a JSON file
            with open("stock_data.json", "w") as f:
                json.dump(stock_data, f, indent=4)
            
            print("Stock data scraped and saved successfully!")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to use scrapegraphai with Gemini API (hypothetical example)
def scrape_with_scrapegraphai():
    try:
        # Hypothetical scrapegraphai integration
        from scrapegraphai import GraphAI
        from scrapegraphai.nodes import GeminiNode

        # Initialize GraphAI with Gemini API
        graph = GraphAI(api_key="your_gemini_api_key")  # Replace with your actual Gemini API key
        gemini_node = GeminiNode(graph)

        # Define the scraping task (hypothetical)
        result = gemini_node.scrape(stock_url, {"selector": "table.stocks"})  # Adjust based on scrapegraphai's API
        
        # Save the result
        with open("stock_data_graphai.json", "w") as f:
            json.dump(result, f, indent=4)
        
        print("Stock data scraped with scrapegraphai and saved successfully!")
    except Exception as e:
        print(f"An error occurred with scrapegraphai: {e}")

# Main execution
if __name__ == "__main__":
    # Scrape using traditional method
    scrape_stock_market()
    
    # Scrape using scrapegraphai (hypothetical)
    scrape_with_scrapegraphai()