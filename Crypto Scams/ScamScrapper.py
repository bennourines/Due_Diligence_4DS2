from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import pandas as pd

website = "https://dfpi.ca.gov/consumers/crypto/crypto-scam-tracker/"

driver = webdriver.Chrome()
driver.get(website)

data = []

try:
    while True:
        # Fixed: Added missing closing parenthesis
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//tbody/tr')))
        
        rows = driver.find_elements(By.XPATH, '//tbody/tr')
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) != 5:
                continue
            
            # Fixed: Ensured proper string closure
            entry = {
                "Primary Subject": cols[0].text,
                "Complaint Narrative": cols[1].text,
                "Scan Type": cols[2].text,
                "Other Subjects": cols[3].text,
                "Website": cols[4].text
            }
            data.append(entry)
        
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, 'a.paginate_button.next')
            if 'disabled' in next_btn.get_attribute('class'):
                print("Reached last page.")
                break
                
            next_btn.click()
            # Fixed: Proper statement separation
            WebDriverWait(driver, 20).until(
                EC.staleness_of(rows[0]))
            
        except (NoSuchElementException, TimeoutException):
            break

finally:
    driver.quit()

df = pd.DataFrame(data)
df.to_csv('crypto_scams.csv', index=False)