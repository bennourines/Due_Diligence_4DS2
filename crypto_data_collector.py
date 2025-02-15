import os
import csv
import logging
import time
import requests
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("CryptoDataCollector")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Custom exception class
class APIError(Exception):
    pass

class Config:
    API_KEYS = {
        'coingecko': os.getenv('COINGECKO_API_KEY')
    }
    
    # CSV file to store market data; defaults to 'market_data.csv' if not set in .env
    CSV_FILE = os.getenv('CSV_FILE', 'market_data.csv')
    
    @classmethod
    def get_rate_limits(cls):
        return {
            'coingecko': 50
        }

class APIClientBase:
    def __init__(self, name):
        self.name = name
        self.base_url = None
        self.rate_limit = Config.get_rate_limits().get(name, 10)
        self.last_request = 0
    
    def rate_limiter(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = 3
            while retries > 0:
                elapsed = time.time() - self.last_request
                if elapsed < 60 / self.rate_limit:
                    time.sleep(60 / self.rate_limit - elapsed)
                try:
                    response = func(self, *args, **kwargs)
                    if response:
                        self.last_request = time.time()
                        return response
                except APIError as e:
                    logger.error(f"API Error: {str(e)}")
                retries -= 1
                time.sleep(2 ** (3 - retries))
            raise APIError("Failed after multiple retries")
        return wrapper

    def handle_errors(self, response):
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            raise APIError(f"API Error: {response.status_code}")

class CoinGeckoClient(APIClientBase):
    def __init__(self):
        super().__init__('coingecko')
        self.base_url = 'https://api.coingecko.com/api/v3/'

    @APIClientBase.rate_limiter
    def get_real_time_prices(self, coin_ids):
        """
        Gets real-time price data for the specified coins.
        coin_ids should be a list of coin IDs as defined in the CoinGecko API (e.g., ['bitcoin', 'ethereum']).
        """
        url = f"{self.base_url}simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd'
        }
        response = requests.get(url, params=params)
        self.handle_errors(response)
        return response.json()

class CryptoDataCollector:
    def __init__(self):
        self.gecko = CoinGeckoClient()
        self.csv_file = Config.CSV_FILE
        # If the CSV file does not exist, create it with a header
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'coin', 'price_usd'])
    
    def collect_all_data(self):
        try:
            # Fetch real-time prices from CoinGecko
            prices = self.gecko.get_real_time_prices(['bitcoin', 'ethereum'])
            processed = self._process_price_data(prices)
            self._write_to_csv(processed)
            logger.info("Data collection cycle completed successfully.")
        except Exception as e:
            logger.error(f"Collection error: {str(e)}")
            raise

    def _process_price_data(self, raw_data):
        processed = []
        now = datetime.utcnow().isoformat()
        for coin, info in raw_data.items():
            processed.append({
                'timestamp': now,
                'coin': coin,
                'price_usd': info.get('usd')
            })
        return processed

    def _write_to_csv(self, data):
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow([row['timestamp'], row['coin'], row['price_usd']])
            logger.info(f"Wrote {len(data)} rows to {self.csv_file}")

# For testing when running this file directly:
if __name__ == "__main__":
    collector = CryptoDataCollector()
    collector.collect_all_data()