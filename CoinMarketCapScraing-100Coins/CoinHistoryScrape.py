import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the Chrome WebDriver
driver = webdriver.Chrome()

# Read coins.csv (should contain columns "Coin Name" and "Coin Symbol")
coins_df = pd.read_csv("coins.csv")

# Iterate over each coin in the CSV file
for idx, row in coins_df.iterrows():
    coin_name = row["Coin Name"]
    coin_symbol = row["Symbol"]
    
    # Convert coin name to a URL-friendly slug (e.g., "Bitcoin Cash" -> "bitcoin-cash")
    coin_slug = coin_name.lower().replace(" ", "-")
    
    # Construct the historical data URL for the coin
    url = f"https://coinmarketcap.com/currencies/{coin_slug}/historical-data/"
    print(f"\nProcessing {coin_name} ({coin_symbol}) at URL: {url}")
    
    # Navigate to the coin's historical data page
    driver.get(url)
    
    # Wait for the table rows to load
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//tbody/tr'))
        )
    except Exception as e:
        print(f"Error waiting for table rows for {coin_name}: {e}")
        continue  # Skip to the next coin if the table doesn't load
    
    # --- Load all historical data by clicking the "Load More" button repeatedly ---
    while True:
        try:
            load_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]"))
            )
            load_more_button.click()
            print("Loaded more data...")
            time.sleep(3)  # Wait for additional rows to load
        except Exception as e:
            # If the button is not found or clickable, assume all data is loaded
            print("All historical data loaded or no 'Load More' button found.")
            break

    # --- Scrape the loaded table data ---
    print("Scraping final dataset...")
    time.sleep(3)  # Optional pause before scraping
    
    data = []  # This will store our scraped data
    table_rows = driver.find_elements(By.TAG_NAME, "tr")
    total_rows = len(table_rows)
    row_count = 0

    print(f"Found {total_rows} rows to process...")
    
    # Loop through each row in the table
    for index, row_elem in enumerate(table_rows, 1):
        columns = row_elem.find_elements(By.TAG_NAME, "td")
        # Skip rows that don't have enough columns (e.g., header rows)
        if len(columns) < 7:
            continue
        
        # Extract the date and other data from the row
        date = columns[0].text.strip()
        open_price = columns[1].text.strip()
        high = columns[2].text.strip()
        low = columns[3].text.strip()
        close = columns[4].text.strip()
        volume = columns[5].text.strip()
        market_cap = columns[6].text.strip()
        
        # Provide visual feedback (updates on the same line)
        print(f"Scraping data for {date}", end='\r')
        
        # Append the row data (with coin metadata) to our list
        data.append([
            coin_name,
            coin_symbol,
            date,
            open_price,
            high,
            low,
            close,
            volume,
            market_cap
        ])
        
        row_count += 1
        # Optionally, print progress every 10 rows
        if index % 10 == 0:
            print(f"Processed {index}/{total_rows} rows | Current date: {date}")
    
    print(f"\nSuccessfully scraped {row_count} data entries for {coin_name}!")
    
    # --- Save the scraped data to a CSV file ---
    df = pd.DataFrame(data, columns=[
        "Coin Name", "Coin Symbol", "Date", "Open", "High", 
        "Low", "Close", "Volume", "Market Cap"
    ])
    output_filename = f"{coin_name}_{coin_symbol}_historical_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")
    
    # Optionally, wait a moment before processing the next coin
    time.sleep(2)

# After processing all coins, close the WebDriver
driver.quit()
print("\nAll coin historical data scraping completed!")
