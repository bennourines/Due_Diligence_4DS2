import datetime
import os
import sys
import json
import time
import logging
#from scheduler import Scheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperMultiGraph
from fadhli_proxy.classes import Proxy
from pydantic import BaseModel, Field
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from the .env file
load_dotenv()

# Retrieve the GEMINI key
gemini_key = os.getenv("GEMINI_KEY")
if gemini_key is None:
    raise Exception("GEMINI_KEY is not set in the environment variables.")
logging.info("Gemini key loaded successfully!")

# Adjust the path to point to the fadhli_proxy directory
fadhli_proxy_path = os.path.abspath(r'fadhli_proxy')
sys.path.append(fadhli_proxy_path)

# Debug: Print sys.path
logging.info("Python Path: %s", sys.path)

# Import fadhli_proxy
try:
    from fadhli_proxy.classes import Proxy
except ImportError as e:
    logging.error("ImportError: %s", e)
    exit(1)

class ExtraDetails(BaseModel):
    # Add any additional fields you might need.
    market_rank: Optional[int] = Field(None, description="Market rank of the coin")
    volume_24h: Optional[float] = Field(None, description="24h volume")
    circulating_supply: Optional[float] = Field(None, description="Circulating supply")
    additional_info: Optional[dict] = Field(None, description="Any other extra details")

# Define data models
class Currency(BaseModel):
    last_hour: float = Field(description="The last hour of the currency")
    last_day: float = Field(description="The last day of the currency")
    last_week: float = Field(description="The last week of the currency")

class Stock(BaseModel):
    name: str = Field(description="The name of the stock coin")
    price: float = Field(description="The price of the stock coin")
    currencies: List[Currency] = Field(description="The currencies of the stock coin")
    market_cap: float = Field(description="The market cap of the stock coin")
    volume: float = Field(description="The volume of the stock coin")
    circulating_supply: float = Field(description="The circulating supply of the stock coin")
    change: float = Field(description="The change of the stock coin")
    extra_details: Optional[ExtraDetails] = Field(None, description="Additional scraped details")

class Coins(BaseModel):
    stocks: List[Stock]

# Initialize proxy
proxy = Proxy(autoRotate=True, maxProxies=10).proxy()[0]

# Define the configuration for the scraping pipeline
graph_config = {
    "llm": {
        "api_key": gemini_key,
        "model": "google_genai/gemini-2.0-flash",
        "timeout": 3600,
        "format": "json",
        "load_kwargs": {
            "server": {
                "proxy": f"http://{proxy}"
            }
        },
        "model_tokens": 8192
    },
    "verbose": True,
    "headless": False,
}

# Updated prompt to indicate scraping all daily data
prompt_text = (
 "Extract all available data for the entire day from the stock market website, "
    "including detailed metrics, historical values, additional statistics, charts, "
    "and any extra information available (such as market rank, 24h volume, circulating supply, etc.). "
    "Additionally, fetch historical data for the past 5 years (from today back to 5 years ago). "
    "Ensure the data includes historical prices, market cap, and trading volume. "
    "Return the output in JSON format following the provided schema."
)

def scrape_url(url: str, output_filename: str):
    """Scrapes a single URL with retries and saves the result to a file."""
    smart_scraper_graph = SmartScraperMultiGraph(
        prompt=prompt_text,
        source=[url],
        schema=Coins,
        config=graph_config
    )

    max_retries = 5
    retry_delay = 10  # seconds

    for attempt in range(1, max_retries + 1):
        logging.info("Scraping %s: Attempt %d of %d", url, attempt, max_retries)
        try:
            result = smart_scraper_graph.run()
            # Validate the result structure
            if result and isinstance(result, dict) and "stocks" in result and result["stocks"]:
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                logging.info("Data from %s saved to %s", url, output_filename)
                return  # Exit after successful scrape
            else:
                logging.warning("Received empty or invalid data from %s. Retrying in %d seconds...", url, retry_delay)
                time.sleep(retry_delay)
        except Exception as e:
            logging.error("Error scraping %s: %s. Retrying in %d seconds...", url, e, retry_delay)
            time.sleep(retry_delay)

    logging.error("Max retries reached for %s. Skipping this URL.", url)
#historical
def scrape_historical_data(url: str, output_filename: str):
    """Scrapes historical stock data for the past 5 years."""
    today = datetime.datetime.today()
    years = [today - datetime.timedelta(days=365 * i) for i in range(6)]  # Last 5 years

    for year in years:
        formatted_date = year.strftime("%Y-%m-%d")
        logging.info("Scraping historical data for %s on %s", url, formatted_date)

        historical_prompt = (
            f"Extract all available historical stock market data from {formatted_date}. "
            "Include price, market cap, trading volume, and any additional statistics. "
            "Return the output in JSON format following the provided schema."
        )

        smart_scraper_graph = SmartScraperMultiGraph(
            prompt=historical_prompt,
            source=[url],
            schema=Coins,
            config=graph_config
        )

        try:
            result = smart_scraper_graph.run()
            if result and isinstance(result, dict) and "stocks" in result and result["stocks"]:
                file_path = f"{output_filename}_{formatted_date}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                logging.info("Historical data from %s saved to %s", url, file_path)
        except Exception as e:
            logging.error("Error scraping historical data from %s for %s: %s", url, formatted_date, e)




def scheduled_scraping():
    """Scheduler function that runs the scraping process every 30 minutes."""
    urls = ["https://coinmarketcap.com/", "https://www.coingecko.com/", "https://www.nasdaq.com/","https://www.investing.com/charts/live-charts","https://www.binance.com/en/markets/overview","https://www.tradingview.com/markets/cryptocurrencies/","https://www.coinbase.com/fr-fr/explore","https://www.kucoin.com/markets"]  # List of URLs to scrape

    # Loop through each URL and scrape data
    for idx, url in enumerate(urls, start=1):
        daily_output_filename = f"crypto_market_data_{idx}.json"
        historical_output_filename = f"crypto_market_historical_{idx}"

        scrape_url(url, daily_output_filename)
        scrape_historical_data(url, historical_output_filename)

# Create a scheduler to run the scraping function every 30 minutes
#scheduler = Scheduler()
scheduler = BlockingScheduler()
scheduler.add_job(scheduled_scraping, 'interval', minutes=30)

def main():
    # Create the scheduler
    scheduler = BlockingScheduler()
    # Schedule the job to run every 30 minutes
    scheduler.add_job(scheduled_scraping, 'interval', minutes=2)

    logging.info("Scheduler started. Scraping will run every 30 minutes.")
    try:
        # Start the scheduler (this will block the main thread)
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")
if __name__ == "__main__":
    main()


#inplace of this 
#result = smart_scraper_graph.run()

#output_file = "crypto_market_data.json"
#with open(output_file, "w", encoding="utf-8") as f:
#    json.dump(result, f, indent=4, ensure_ascii=False)