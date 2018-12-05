import sys
from datetime import timezone
import pandas as pd
import numpy as np
import pickle

from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences

ticker = "AAPL"

time_shift = 4
# News and stock price file path
news_path = './db/' + ticker + '_clean_news_20170601_20180701.p'
price_path = './db/' + ticker + '_stock_price_20170601_20180701.p'
# Vocab, model file path
vocab_index_path = './model/word_idx.p'
vocab_path = './model/final_vocab.p'
model_path = './model/cnn_best_model.h5'


def createDataSet(news, pct):
    x_data = []
    y_data = []
    date_data = []
    for _news in news:
        date_data.append(_news[0])
        x_data.append(_news[1])
        if pct[ticker][_news[0]] >= 0:
            y_data.append(1)
        else:
            y_data.append(0)
    assert len(x_data) == len(y_data)
    return np.array(x_data), np.array(y_data), np.array(date_data)


def vectorize_sentences(data, word_idx, final_vocab, maxlen=40):
    X = []
    paddingIdx = len(final_vocab) + 2
    for sentences in data:
        x = []
        for word in sentences:
            if word in final_vocab:
                x.append(word_idx[word])
            elif word.lower() in final_vocab:
                x.append(word_idx[word.lower()])
            else:
                x.append(paddingIdx)
        X.append(x)
    return pad_sequences(X, maxlen=maxlen)


def readData():
    news = pickle.load(open(news_path, 'rb'))
    prices = pickle.load(open(price_path, 'rb'))

    _date_list = [x[0] for x in prices]
    prices_list = [x[1] for x in prices]
    df = pd.DataFrame({
        ticker: prices_list},
        index=_date_list)
    df_pct = df.pct_change(periods=time_shift)

    return news, prices, df_pct


def main():
    news, prices, df_pct = readData()
    print('====================================')
    print('Finish reading news and prices')
    sys.stdout.flush()
    x_data, y_data, date_data = createDataSet(news, df_pct)
    # x_count = [len(word_tokenize(x)) for x in x_data]

    maxlen = 150
    Xs = []
    for sentence in x_data:
        Xs.append(word_tokenize(sentence))
    word_idx = pickle.load(open(vocab_index_path, 'rb'))
    final_vocab = pickle.load(open(vocab_path, 'rb'))
    from keras.models import load_model
    model = load_model(model_path)
    train_data = vectorize_sentences(Xs, word_idx, final_vocab, maxlen=maxlen)
    classes = model.predict(train_data, batch_size=32)

    from sklearn import metrics
    from sklearn.metrics import confusion_matrix, accuracy_score
    expected = y_data
    predicted = classes.argmax(axis=-1)
    report = metrics.classification_report(expected, predicted)
    print(confusion_matrix(expected, predicted))
    print(report)
    print("Accuracy: %s" % accuracy_score(expected, predicted))
    print('====================================')

    predicted = classes
    date_score = []
    date_list = list(set(date_data))
    for date in date_list:
        score = 0
        counter = 0
        for i in range(len(predicted)):
            if date_data[i] == date:
                score += predicted[i][1]
                counter += 1
        assert counter != 0, 'division by zero !!!'
        score = float(score) / float(counter)
        date_score.append([date, score, counter])

    _expected = []
    _predicted = []
    for data in date_score:
        if data[1] >= 0.5:
            _predicted.append(1)
        else:
            _predicted.append(0)
        if df_pct[ticker][data[0]] >= 0:
            _expected.append(1)
        else:
            _expected.append(0)

    report = metrics.classification_report(_expected, _predicted)
    print(confusion_matrix(_expected, _predicted))
    print(report)
    print("Accuracy: %s" % accuracy_score(_expected, _predicted))

    import matplotlib
    from matplotlib import pyplot
    score_list = [x[1] for x in date_score]
    pyplot.hist(score_list)

    date_score = sorted(date_score, key=lambda x: x[0])
    news_date_list, news_score_list, counter_list = zip(*date_score)
    _date_list, prices_list = zip(*df_pct[ticker].items())
    # _date_list = _date_list[-365:]
    # prices_list = prices_list[-365:]
    _date_list = [x.date() for x in _date_list]
    font = {'size': 8}
    matplotlib.rc('font', **font)
    fig, ax = pyplot.subplots()
    ax.plot(_date_list, prices_list, "#1f77b4", label="Stock Price")
    ax2 = ax.twinx()
    ax2.plot(news_date_list, news_score_list,
             "#ff7f0e", label="Sentiment Score")
    # _date_score = [[int(x[0].timestamp()*1000),
    #                x[1], x[2]] for x in date_score]
    # _date_score = sorted(_date_score, key=lambda x: x[0])
    # pickle.dump(_date_score, open('date_score.p', 'wb'))


if __name__ == '__main__':
    main()
