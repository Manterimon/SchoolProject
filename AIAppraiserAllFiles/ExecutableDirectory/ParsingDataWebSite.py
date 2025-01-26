from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas

class ParserFunPay:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.get("https://funpay.com/lots/436/?ysclid=m3lbz9gxox531119583")
    def startParsFunPay(self):
        elements = WebDriverWait(self.driver, 5).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".tc-item"))
        )
        df = pandas.DataFrame(columns=['Price', 'Cup', 'Hero'])
        i = 0
        for element in elements:
            price_element = element.find_element(By.CSS_SELECTOR, ".tc-price")
            self.attr_value_cup = element.get_attribute('data-f-cup')
            self.attr_value_hero = element.get_attribute('data-f-hero')
            self.attr_value_price = price_element.get_attribute('data-s')
            df = df._append({'Cup': self.attr_value_cup, 'Hero': self.attr_value_hero, 'Price': self.attr_value_price}, ignore_index=True)
            i +=1
            print(f'Cup: {self.attr_value_cup}, Person: {self.attr_value_hero}, Price data: {self.attr_value_price}')
        print(i)
        return df

# class ParserPlayerok:
#     def __init__(self):
#         self.driver = webdriver.Chrome()
#         self.driver.get('https://paygame.ru/games/brawl-stars/offers?type=account&ysclid=m4efywib7v444073887')
#
#     def startParsPlayerok(self):
#         elements = WebDriverWait(self.driver, 5).until(
#             EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".sc-17v71la-2.fYJfba"))
#         )
#         for element in elements:
#             sub_element = element.find_element(By.CSS_SELECTOR, ".sc-17v71la-4.lpbdul")
#             inner_element = sub_element.find_element(By.CSS_SELECTOR, ".sc-17v71la-5.kSaaSH")
#             dd_element = inner_element.find_element(By.CSS_SELECTOR, '.sc-17v71la-9.dCzdPA')
#
#             price_element = inner_element.find_element(By.CSS_SELECTOR, ".sc-17v71la-16.PFkwM")
#
#             hero_element = inner_element.find_element(By.CSS_SELECTOR, ".sc-17v71la-12.dEiLuI")
#             cup_element = inner_element.find_element(By.CSS_SELECTOR, '.sc-17v71la-12.dEiLuI')
#
#             cup_text = cup_element.text
#             hero_text = hero_element.text
#             price_text = price_element.text.replace("\u00a0", "").replace(' ', '').replace('₽', '')  # Удаление &nbsp;
#             print(f'Price: {price_text}, Hero: {hero_text}, Cup: {cup_text}')


# Basic Flask App Setup
def databaseDivision(df):
    mid_index = len(df) // 2
    first_half = df.iloc[:mid_index]
    last_half = df.iloc[mid_index:]
    return first_half, last_half

if __name__ == '__main__':
    df = pandas.read_csv('DataParsFunPay.csv')
    first_half, last_half = databaseDivision(df)
    first_half.to_csv('first_DFFP.csv', index=False)
    last_half.to_csv('last_DFFP.csv', index=False)
    print(df)