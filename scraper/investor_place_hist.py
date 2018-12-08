import requests
from bs4 import BeautifulSoup
import datetime
from pymongo import MongoClient

import time
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

# from django.utils import encoding

client = MongoClient('ec2-13-229-79-3.ap-southeast-1.compute.amazonaws.com', username='di_user',
                     password='u32nODdnipIl9ZAmHpQ2uLMsUY3lbT9', authSource='admin', authMechanism='SCRAM-SHA-1')


def loadTicker():
    _tickers = []
    with open("../data/sp500.txt") as f:
        for line in f:
            _tickers.append(line.rstrip('\n'))
    _tickers = _tickers[229:]
    return _tickers


def insertoDB(data):
    global client
    print("Inserting to database ...")
    sys.stdout.flush()

    db = client.news
    stock = db.investor_place

    j = 0
    for _date in data:
        try:
            stock.insert_one(_date)
            j += 1
        except Exception as e:
            print(e)
    print("Inserted ", str(j), " news to database !!")


def main():
    chrome_path = '../chromedriver_win32/chromedriver'

    headers = {
        'User-Agent': 'My User Agent 1.0',
        # 'From': 'youremail@domain.com'  # This is another valid field
    }
    root_url = 'https://investorplace.com'
    base_url = "/stock-quotes/"
    # _ticker = 'AAPL'
    extended_url = "-stock-quote/"

    driver = webdriver.Chrome(chrome_path)  # webdriver.Firefox() in your case

    tickers = loadTicker()
    for _ticker in tickers:
        url = root_url + base_url + _ticker + extended_url
        driver.get(url)
        try:
            for i in range(0, 15):
                driver.find_element_by_css_selector('#ipm_article_load_more').click()
                print("Clicking button", str(i))
                time.sleep(5)
        except Exception as e:
            print(e)
        time.sleep(5)

        try:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            id_list = [x['id'] for x in soup.select('li[id*=post]')]
            url_list = [x['href'] for x in soup.select('h4.entry-title a')]
            title_list = [x['title'] for x in soup.select('h4.entry-title a')]

            if len(id_list) != len(url_list):
                print('error !!!', _ticker)
                continue
        except Exception as e:
            print(str(e), url)
            continue

        news = []
        for id, _url, headline in list(zip(id_list, url_list, title_list)):
            try:
                print("Visiting", _url)
                id = re.findall('\d+', id)
                id = id[0]
                resp = requests.get(_url, headers=headers)
                soup = BeautifulSoup(resp.text, 'html.parser')
                str_date = soup.select('meta[itemprop=datePublished]')[0]['content'][:19]
                date = datetime.datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S')
                paragraphs = soup.select('.entry-content p')
                text = ''
                for paragraph in paragraphs[:len(paragraphs) - 1]:
                    text += paragraph.text
                news.append({"_id": id + _ticker, "headline": headline, "text": text, "date": date, "ticker": _ticker})
            except Exception as e:
                print(str(e), _url)

        insertoDB(news)
    # driver.close()


if __name__ == '__main__':
    main()
