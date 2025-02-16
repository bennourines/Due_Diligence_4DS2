from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd

# Initialize the WebDriver
driver = webdriver.Chrome()

try:
    # Open the website
    website = 'https://bitscreener.com/?t=listview&p=1'
    driver.get(website)

    # Remove the ad that might block clicks
    driver.execute_script("var ad = document.getElementById('react-fixed-ads'); if (ad) ad.remove();")

    # Wait for the dropdown (show-more-select) to be clickable
    dropdown = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.show-more-select[tabindex="0"]'))
    )
    print("Dropdown is clickable.")

    # Scroll to the dropdown before clicking
    driver.execute_script("arguments[0].scrollIntoView();", dropdown)
    ActionChains(driver).move_to_element(dropdown).click().perform()
    print("Clicked the dropdown.")

    # Wait for the <a> element with text '500' to be visible
    element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, "//a[text()='500']"))
    )
    print("Found the <a> element with text '500'.")

    # Click the <a> element using JavaScript to avoid click interception
    driver.execute_script("arguments[0].click();", element)
    print("Clicked the <a> element with text '500'.")

    # Wait for the table to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'tr'))
    )
    print("Table loaded.")

    # List to store scraped data
    data = []

    # Function to scrape data from the current page
    def scrape_page():
        rows = driver.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            columns = row.find_elements(By.TAG_NAME, "td")
            if len(columns) < 10:  # Adjust this number based on your table structure
                continue
            rank = columns[0].text.strip()
            coin_name = columns[1].text.strip()
            market_cap = columns[2].text.strip()
            price = columns[3].text.strip()
            volume_24h = columns[4].text.strip()
            change_1h = columns[6].text.strip()
            change_24h = columns[7].text.strip()
            change_7d = columns[8].text.strip()
            change_1y = columns[9].text.strip()
            print(f"Rank: {rank}")
            print(f"Coin Name: {coin_name}")
            print(f"Market Cap: {market_cap}")
            print(f"Price: {price}")
            print(f"Volume (24h): {volume_24h}")
            print(f"1h Change: {change_1h}")
            print(f"24h Change: {change_24h}")
            print(f"7d Change: {change_7d}")
            print(f"1y Change: {change_1y}")
            print("-" * 40)
            # Append data to the list
            data.append([rank, coin_name, market_cap, price, volume_24h, change_1h, change_24h, change_7d, change_1y])

    # Scrape the first page
    scrape_page()

    # Loop through the pagination
    for page in range(2, 51):  # Loop from page 2 to page 50
        try:
            # Find the pagination link for the current page
            pagination_link = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//a[@href='?t=listview&p={page}']"))
            )
            print(f"Found pagination link for page {page}.")

            # Click the pagination link using JavaScript to avoid interception
            driver.execute_script("arguments[0].click();", pagination_link)
            print(f"Clicked pagination link for page {page}.")

            # Wait for the table to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'tr'))
            )
            print(f"Table loaded for page {page}.")

            # Scrape the current page
            scrape_page()

        except Exception as e:
            print(f"An error occurred on page {page}: {e}")
            break

    # After scraping all pages, save data to a CSV file
    df = pd.DataFrame(data, columns=["Rank", "Coin Name", "Market Cap", "Price", "Volume (24h)", "1h Change", "24h Change", "7d Change", "1y Change"])
    df.to_csv("crypto_data.csv", index=False)
    print("Data saved to crypto_data.csv.")

except Exception as e:
    print("An error occurred:", e)

finally:
    # Wait for user input before closing the browser
    input("Press Enter to close the browser...")
    driver.quit()
