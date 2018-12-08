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
    page_counter += 10
    page_lock.release()
    return tmp


def loadTicker():
    db = client.news
    sp500 = db.sp500
    _tickers = []
    for ticker in sp500.find():
        _tickers.append(ticker['ticker'])
    _tickers = _tickers[312:]
    return _tickers


def insertoDB(data):
    global client
    print("Inserting to database ...")
    sys.stdout.flush()

    db = client.news
    stock = db.investors

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
    root_url = 'https://services.investors.com'
    base_url = "/searchapi/searchresults?&ntt="
    # _ticker = 'AAPL'
    page = "&no="
    extended_url = "&nr=&ns=&n=&nf=&more=&dym=&source=&module="

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

        url = root_url + base_url + _ticker + page + str(page_number) + extended_url
        print('Visiting page', url)
        sys.stdout.flush()
        resp = requests.get(url, headers=headers)
        xml_resp = resp.json()

        title_url_list = [(x['Title'], x['Url']) for x in xml_resp['Results']]

        if len(title_url_list) == 0:
            exit_lock.acquire()
            exit_counter += 1
            exit_lock.release()
            continue

        for headline, _url in title_url_list:
            resp = requests.get(_url, headers=headers)
            soup = BeautifulSoup(resp.text, 'html.parser')
            try:
                text = ''
                for paragraph in soup.select("div.single-post-content p"):
                    if 'related' in paragraph.text.lower():
                        break
                    else:
                        text += paragraph.text
                text = text.replace('\n', '')
                text = text.lstrip()
                if not text:
                    continue
                meta = soup.select('meta[property=article:published_time]')[0]
                date = meta['content'][:19]
                date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')

                linktag = soup.select("link[rel=shortlink]")[0]
                short_link = linktag['href']

                id = re.findall('\d+', short_link)[0]
                news.append({"_id": id+_ticker, "headline": headline, "text": text, "date": date, "ticker": _ticker})
            except Exception as e:
                # print(str(e), _url)
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
    min_pages = 2000
    no_of_thread = 20
    for ticker in tickers:
        for i in range(no_of_thread):
            sleep(1)
            try:
                thread = threading.Thread(target=main, args=(ticker, min_pages))
                thread.start()
                thread_list.append(thread)
            except:
                print("Error: unable to start thread")
        for t in thread_list:
            t.join()
        print("Inserted", str(global_counter), "news for", ticker, "to database totally!!")
        page_counter = 0
        global_counter = 0
        exit_counter = 0
        thread_list.clear()
