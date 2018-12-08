import requests
from bs4 import BeautifulSoup
import datetime
from pymongo import MongoClient

import time
import threading
from threading import Lock
import sys
import re
from time import sleep

import configparser
config = configparser.ConfigParser()
config.read('mongo_user_info.ini')
mongo = config['mongo']
client = MongoClient(mongo['host'], username=mongo['username'],
                     password=mongo['password'], authSource=mongo['authSource'], authMechanism=mongo['authMechanism'])
lock = Lock()
database_lock = Lock()
page_lock = Lock()
exit_lock = Lock()
page_counter = 0
exit_counter = 0
global_counter = 0


def getPageResource():
    global page_counter
    page_lock.acquire()
    tmp = page_counter
    page_counter += 1
    page_lock.release()
    return tmp


def loadTicker():
    db = client.news
    sp500 = db.sp500
    _tickers = []
    for ticker in sp500.find():
        # if ticker['ticker'] == 'AAPL':
        #     continue
        _tickers.append(ticker['ticker'])
    _tickers = _tickers[498:]
    return _tickers


def insertoDB(data):
    global file, client
    print("Inserting to database ...")
    sys.stdout.flush()

    db = client.news
    stock = db.thestreet

    j = 0
    for _data in data:
        try:
            stock.insert_one(_data)
            j += 1
        except Exception as e:
            print(e)
    print("Inserted ", str(j), " news to database !!")
    return j


def main(_ticker, _min_pages):
    global global_counter, page_counter, exit_counter
    start_time = time.time()
    root_url = 'https://www.thestreet.com'
    base_url = "/quote/"
    # stock = 'AAPL'
    extended_url = "/details/news"
    page = "?page="

    print(datetime.datetime.now())
    print("Getting urls ....")
    sys.stdout.flush()

    headers = {
        'User-Agent': 'My User Agent 1.0',
        # 'From': 'youremail@domain.com'  # This is another valid field
    }

    print("Searching for", _ticker)
    sys.stdout.flush()
    while page_counter <= _min_pages or exit_counter <= 3:
        page_number = getPageResource()
        news = []
        url_date_list = []

        if page_number == 0:
            url = root_url + base_url + _ticker + extended_url
        else:
            url = root_url + base_url + _ticker + extended_url + page + str(page_number)
        print('Visiting page', url)
        sys.stdout.flush()
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')

        array = soup.select("div.news-list-compact__body.columnRight > a[href]")
        raw_date_list = soup.select('.news-list__publish-date time')

        if len(array) == 0:
            exit_lock.acquire()
            exit_counter += 1
            exit_lock.release()
            continue

        if len(array) != len(raw_date_list):
            print("error !!!")
            sys.stdout.flush()
            continue

        for item, date in list(zip(array, raw_date_list)):
            d = datetime.datetime.strptime(date['datetime'], '%Y-%m-%dT%H:%MZ')
            url_date_list.append((item.get("href"), d))

        for link, date in url_date_list:
            if link[:4] != 'http':
                _url = root_url + link
            else:
                _url = link
            resp = requests.get(_url, headers=headers)
            soup = BeautifulSoup(resp.text, 'html.parser')
            try:
                headline = soup.find("h1", "article__headline ").text
                text = soup.find('div', 'article__body ').text
                text = text.replace('\n', '')
                text = text.lstrip()
                if not text:
                    continue
                id = soup.find("article", "article")['id']
                id = re.findall('\d+', id)[0]
                news.append({"_id": id+_ticker, "headline": headline, "text": text, "date": date, "ticker": _ticker})
            except Exception as e:
                # print(_url + "\t" + str(e))
                continue
        database_lock.acquire()
        counter = insertoDB(news)
        database_lock.release()

        lock.acquire()
        global_counter += counter
        lock.release()
    print("--- %s seconds ---" % (time.time() - start_time))
    sys.stdout.flush()


if __name__ == '__main__':
    tickers = loadTicker()
    thread_list = []
    min_pages = 400
    no_of_thread = 20
    for ticker in tickers:
        for i in range(no_of_thread+1):
            sleep(1)
            try:
                thread = threading.Thread(target=main, args=(ticker, min_pages))
                thread.start()
                thread_list.append(thread)
            except:
                print("Error: unable to start thread")
        for t in thread_list:
            t.join()
        print("Inserted", str(global_counter), "news for", ticker + "to database totally!!")
        page_counter = 0
        global_counter = 0
        exit_counter = 0
