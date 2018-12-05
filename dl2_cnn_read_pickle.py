import sys
import pandas as pd
import numpy as np
import pickle

from nltk import word_tokenize
import operator
from functools import reduce
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

ticker = "AAPL"
# {'AAPL', 'AMZN', 'MSFT', 'BAC', 'C', 'GOOG', 'JPM', 'INTC', 'GE', 'WMT' }
time_shift = 4
seed = 10
# 30, 33, 61
# News and stock price file path
news_path = './db/' + ticker + '_clean_news_20170601_20180701.p'
price_path = './db/' + ticker + '_stock_price_20170601_20180701.p'


def unison_shuffled_copies(a, b):
    np.random.seed(seed)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def createDataSet(news, pct, shuffle=True):
    x_data = []
    y_data = []
    for _news in news:
        x_data.append(_news[1])
        if pct[ticker][_news[0]] >= 0:
            y_data.append(1)
        else:
            y_data.append(0)
    assert len(x_data) == len(y_data)
    if shuffle:
        return unison_shuffled_copies(np.array(x_data), np.array(y_data))
    else:
        return np.array(x_data), np.array(y_data)


def word_freq(Xs, num):
    all_words = [words.lower() for sentences in Xs for words in sentences]
    sorted_vocab = sorted(dict(Counter(all_words)).items(), key=operator.itemgetter(1))
    final_vocab = [k for k, v in sorted_vocab if v > num]
    word_idx = dict((c, i + 1) for i, c in enumerate(final_vocab))
    pickle.dump(word_idx, open('./model/' + '/word_idx.p', 'wb'))
    pickle.dump(final_vocab, open('./model/' + '/final_vocab.p', 'wb'))

    return final_vocab, word_idx


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
    # from matplotlib import pyplot
    # pyplot.hist(df_pct['AAPL'].tolist()[4:])
    return news, prices, df_pct


def main():
    news, prices, df_pct = readData()
    print('====================================')
    print('Random Seed: %d' % seed)
    print('Finish reading news and prices')
    sys.stdout.flush()
    x_data, y_data = createDataSet(news, df_pct, shuffle=True)
    x_count = [len(word_tokenize(x)) for x in x_data]

    maxlen = 150
    EMBEDDING_DIM = 100  # 200, 300
    filters = 128
    hidden_dims = 100
    batch_size = 128
    epochs = 30

    Xs = []
    for sentence in x_data:
        Xs.append(word_tokenize(sentence))
    vocab = sorted(reduce(lambda x, y: x | y, (set(words) for words in Xs)))
    final_vocab, word_idx = word_freq(Xs, 2)
    vocab_len = len(final_vocab)
    train_data = vectorize_sentences(Xs, word_idx, final_vocab, maxlen=maxlen)

    from keras.layers import Conv1D, Input, Concatenate, MaxPooling1D
    from keras.layers.core import Activation, Flatten, Dropout, Dense
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential, Model
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras import regularizers

    # filter_sizes = (2, 4, 5, 8)
    filter_sizes = (2, 3, 4, 5)
    dropout_prob = [0.4, 0.5]
    n_in = maxlen

    graph_in = Input(shape=(maxlen, EMBEDDING_DIM))
    convs = []
    for fsz in filter_sizes:
        # 256
        conv = Conv1D(128,
                      fsz,
                      padding='valid',
                      activation='relu',
                      strides=1)(graph_in)
        pool = MaxPooling1D(pool_size=n_in - fsz + 1)(conv)
        flattenMax = Flatten()(pool)
        convs.append(flattenMax)

    if len(filter_sizes) > 1:
        out = Concatenate(axis=-1)(convs)
    else:
        out = convs[0]
    graph = Model(inputs=graph_in, outputs=out, name="graphModel")

    model = Sequential()
    model.add(Embedding(vocab_len + 3,  # size of vocabulary
                        EMBEDDING_DIM,
                        input_length=maxlen))
    # model.add(Dropout(dropout_prob[0]))
    model.add(graph)
    # model.add(Dense(1024)) # 512
    model.add(Dense(256))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    optimizer = Adam(0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    _y_data = to_categorical(y_data)
    up = y_data.tolist().count(1)
    down = y_data.tolist().count(0)

    split = int(len(train_data) * 0.1)
    x_train = train_data[:-split]
    y_train = _y_data[:-split]
    x_test = train_data[-split:]
    y_test = _y_data[-split:]
    stop = EarlyStopping(patience=3)
    checkpoint = ModelCheckpoint('cnn_best_model.h5',
                                 verbose=0,
                                 monitor='val_acc',
                                 save_best_only=True)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        epochs=epochs, callbacks=[stop, checkpoint],
                        verbose=2)
    classes = model.predict(x_test, batch_size=128)

    from sklearn import metrics
    from sklearn.metrics import confusion_matrix, accuracy_score
    expected = y_test.argmax(axis=-1)
    predicted = classes.argmax(axis=-1)
    report = metrics.classification_report(expected, predicted)
    print(confusion_matrix(expected, predicted))
    print(report)
    print("Accuracy: %s" % accuracy_score(expected, predicted))
    print('====================================')
    '''
    from matplotlib import pyplot
    pyplot.subplot(2, 1, 1)
    pyplot.plot(history.history['val_loss'])
    pyplot.subplot(2, 1, 2)
    pyplot.plot(history.history['val_acc'])
    pyplot.show()
    '''

    # from keras.models import load_model
    # model.save('cnn_model.h5')
    # del model
    # model = load_model('my_model.h5')


if __name__ == '__main__':
    main()
