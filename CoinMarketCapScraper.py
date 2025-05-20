import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd 

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the website
website = 'https://coinmarketcap.com/'
driver.get(website)

# Remove ads
driver.execute_script("var ad = document.getElementById('react-fixed-ads'); if (ad) ad.remove();")

try:
    # Wait for the coin name and symbol elements to be present in the DOM
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CLASS_NAME, "coin-item-name"))
    )
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "coin-item-symbol"))
    )

    # Extract coin name and symbol
    coin_name = driver.find_element(By.CLASS_NAME, "coin-item-name").text
    coin_symbol = driver.find_element(By.CLASS_NAME, "coin-item-symbol").text

    print(f"Coin Name: {coin_name}")
    print(f"Coin Symbol: {coin_symbol}")

    # Navigate to historical data page
    parent_div = driver.find_element(By.CLASS_NAME, "sc-65e7f566-0.sc-e8147118-2.eQBACe.kHBMUJ")
    parent_div.click()
    time.sleep(5)

    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/currencies/bitcoin/historical-data/') and contains(text(), 'See historical data')]"))
    )
    historical_data_link = driver.find_element(By.XPATH, "//a[contains(@href, '/currencies/bitcoin/historical-data/') and contains(text(), 'See historical data')]")
    driver.execute_script("arguments[0].scrollIntoView(true);", historical_data_link)
    historical_data_link.click()
    time.sleep(5)

    # Load all historical data
    while True:
        try:
            # Wait for button but don't treat absence as error
            load_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]"))
            )
            load_more_button.click()
            print("Loaded more data...")
            time.sleep(3)  # Reduced sleep time for faster loading
        except Exception as e:
            # Check if it's a "not found" error
            if "NoSuchElementException" in str(e.__class__) or "TimeoutException" in str(e.__class__):
                print("\nAll historical data loaded successfully")
            else:
                print(f"\nUnexpected error occurred: {str(e)}")
            break
    
    # Now scrape all loaded data
    print("\nScraping final dataset...")
    time.sleep(3)

    # Scrape table data with coin name and symbol
    print("\nStarting data scraping...")
    data = []
    table_rows = driver.find_elements(By.TAG_NAME, "tr")
    total_rows = len(table_rows)
    row_count = 0

    print(f"Found {total_rows} rows to process...")
    
    for index, row in enumerate(table_rows, 1):
        columns = row.find_elements(By.TAG_NAME, "td")
        if len(columns) < 7:
            continue
        
        # Extract data with visual feedback
        date = columns[0].text.strip()
        print(f"Scraping data for {date}", end='\r')  # Update current line
        
        data.append([
            coin_name,
            coin_symbol,
            date,
            columns[1].text.strip(),  # Open
            columns[2].text.strip(),  # High
            columns[3].text.strip(),  # Low
            columns[4].text.strip(),  # Close
            columns[5].text.strip(),  # Volume
            columns[6].text.strip()  # Market Cap
        ])
        
        row_count += 1
        # Print progress every 10 rows
        if index % 10 == 0:
            print(f"Processed {index}/{total_rows} rows | Current date: {date}")

    # Final summary
    print(f"\nSuccessfully scraped {row_count} data entries!")
    print("Sample of scraped data:")
    if len(data) > 0:
        print(f"First entry: {data[0][2]} - Open: {data[0][3]}")
        print(f"Last entry: {data[-1][2]} - Open: {data[-1][3]}")
    else:
        print("No data found in table")

    # Save to CSV with additional columns
    df = pd.DataFrame(data, columns=[
        "Coin Name", "Coin Symbol", "Date", "Open", "High", 
        "Low", "Close", "Volume", "Market Cap"
    ])
    df.to_csv(f"{coin_name}_{coin_symbol}_historical_data.csv", index=False)
    print("Data saved with coin metadata!")

    # Redirect back to CoinMarketCap homepage
    driver.get(website)  # <-- ADD THIS LINE
    print("\nRedirected back to CoinMarketCap homepage")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    input("Press Enter to close the browser...")
    driver.quit()