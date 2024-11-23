from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas

class ParserFunPay:
    def __init__(self):
        self.driver = webdriver.Edge()
        self.driver = webdriver.Edge()
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



if __name__ == '__main__':
    pfr = ParserFunPay()
    df = pfr.startParsFunPay()
    df.to_csv('DataParsFunPay.csv', index=False)
    print(df)