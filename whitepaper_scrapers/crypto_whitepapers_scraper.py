import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ------------------- Configuration and Setup -------------------
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Chrome to download files to the current directory
options = webdriver.ChromeOptions()
prefs = {"download.default_directory": os.getcwd()}
options.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(options=options)

# Base URL for the whitepaper page
website = 'https://bitscreener.com/whitepaper'
driver.get(website)

# Remove potential interfering ad container
driver.execute_script("var ad = document.getElementById('react-fixed-ads'); if (ad) ad.remove();")

# Set up an explicit wait
wait = WebDriverWait(driver, 30)

# ------------------- Function to Process Rows on a Page -------------------
def process_current_page(page_num):
    logging.info(f"Processing page {page_num} ...")
    # Wait until table rows are present on the page
    wait.until(EC.presence_of_element_located((By.XPATH, '//tbody//tr')))
    rows = driver.find_elements(By.XPATH, '//tbody//tr')
    total_rows = len(rows)
    logging.info(f"Found {total_rows} rows on page {page_num}.")

    for i in range(total_rows):
        try:
            # Re-fetch rows to avoid stale element errors
            rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//tbody//tr')))
            if i >= len(rows):
                break

            row = rows[i]
            original_url = driver.current_url

            # Locate the download button within the row and scroll it into view
            download_btn = row.find_element(By.XPATH, './/div[contains(@class, "download")]')
            driver.execute_script("arguments[0].scrollIntoView(true);", download_btn)

            # Attempt to click the download button; retry if click is intercepted
            attempt = 0
            clicked = False
            while attempt < 2 and not clicked:
                try:
                    download_btn.click()
                    logging.info(f"Page {page_num}, Row {i+1}: Clicked download button on attempt {attempt+1}.")
                    clicked = True
                except Exception as click_ex:
                    if "element click intercepted" in str(click_ex):
                        logging.warning(f"Page {page_num}, Row {i+1}: Click intercepted on attempt {attempt+1}. Removing interfering iframes.")
                        # Remove interfering iframes (commonly ad iframes with IDs starting with 'aswift_')
                        driver.execute_script(
                            "document.querySelectorAll('iframe[id^=\"aswift_\"]').forEach(el => el.remove());"
                        )
                        time.sleep(1)
                        attempt += 1
                    else:
                        raise click_ex

            if not clicked:
                logging.warning(f"Page {page_num}, Row {i+1}: Could not click download button after retries. Skipping row.")
                continue

            # Wait briefly for the download to initiate or for a redirection to occur
            time.sleep(3)
            new_url = driver.current_url

            # If the click caused a redirection, navigate back and skip the row
            if new_url != original_url:
                logging.info(f"Page {page_num}, Row {i+1}: Redirection detected (URL changed to {new_url}). Navigating back.")
                driver.back()
                time.sleep(3)
                continue
            else:
                logging.info(f"Page {page_num}, Row {i+1}: Download initiated (no redirection).")
                # Wait longer if needed for the download process
                time.sleep(6)

        except Exception as e:
            logging.warning(f"Page {page_num}, Row {i+1}: Exception encountered: {e}. Skipping to next row.")
            continue

# ------------------- Main Pagination Loop -------------------
try:
    # Process the first page
    process_current_page(page_num=1)

    # Loop through subsequent pages (adjust the range as needed)
    max_page = 50  # For example, process pages 2 through 50
    for page in range(2, max_page + 1):
        try:
            # Adjust the XPath below to match your pagination structure.
            # For example, if the link contains 'p=2' (or similar) use contains() to be flexible.
            pagination_xpath = f"//a[contains(@href, 'p={page}')]"
            pagination_link = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, pagination_xpath))
            )
            logging.info(f"Found pagination link for page {page}.")

            # Click the pagination link via JavaScript to avoid interception issues
            driver.execute_script("arguments[0].click();", pagination_link)
            logging.info(f"Clicked pagination link for page {page}.")

            # Wait for the new page’s table to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//tbody//tr'))
            )
            # Process the rows on the current page
            process_current_page(page_num=page)

        except Exception as e:
            logging.info(f"An error occurred on page {page}: {e}. Skipping this page and continuing.")
            continue

    logging.info("Finished processing all pages.")

except Exception as main_e:
    logging.error(f"Unexpected error: {main_e}")

finally:
    input("Press Enter to exit and close the browser...")
    driver.quit()
