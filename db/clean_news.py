import sys
import datetime
import pickle

from string import punctuation
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

ticker = "AAPL"
# {'AAPL', 'AMZN', 'MSFT', 'BAC', 'C', 'GOOG', 'JPM', 'INTC', 'GE', 'WMT' }

news_path = ticker + '_news_20170601_20180701.p'
clean_news_path = ticker + '_clean_news_20170601_20180701.p'


def get_wordnet_pos(treebank_tag):
    if treebank_tag == 'VBG':
        return wordnet.NOUN
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def removeDuplicate(a):
    b_set = set(map(tuple, a))
    b = list(map(list, b_set))
    return b


def noSpecificPhrase(sentence):
    if 'The watch was made in conjunction' in sentence or \
            'Hot On TheStreet' in sentence or \
            'Want to be alerted' in sentence:
        return False
    else:
        return True


def clean_doc(doc):
    sent_list = []
    sentences = sent_tokenize(doc)
    for i in range(len(sentences)):
        if ticker in sentences[i]:
            # or 'apple' in sentences[i].lower():
            if noSpecificPhrase(sentences[i]):
                sent_list.append(sentences[i])
            if i >= 0 and noSpecificPhrase(sentences[i - 1]):
                sent_list.append(sentences[i - 1])
            if i < len(sentences) - 1 and noSpecificPhrase(sentences[i - 1]):
                sent_list.append(sentences[i + 1])

    filtered_sent_list = []
    sent_list = list(set(sent_list))
    stop_words = set(stopwords.words('english'))
    stop_words.remove('up')
    stop_words.remove('down')
    stop_words.remove('below')
    stop_words.remove('won')
    stop_words.remove('further')
    stop_words.remove('most')
    stop_words.remove('few')
    stop_words.remove('after')
    stop_words.remove('against')
    stop_words.remove('own')
    table = str.maketrans('', '', punctuation)
    wordnet_lemmatizer = WordNetLemmatizer()
    for sent in sent_list:
        # Tokenize the sentence
        tokens = word_tokenize(sent)
        # POS tag
        tokens, pos = zip(*pos_tag(tokens))
        # Change to wordnet POS tag
        pos = [get_wordnet_pos(x) for x in pos]
        # Lemmatize
        tokens = [wordnet_lemmatizer.lemmatize(word, pos=pos) for (word, pos) in zip(tokens, pos)]
        # Remove punctuation
        tokens = [w.translate(table) for w in tokens]
        # Remove not alphabetic word
        tokens = [word for word in tokens if word.isalpha()]
        # Remove stop words
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if len(word) > 2]
        tokens = ' '.join(tokens)
        filtered_sent_list.append(tokens)
    return ' '.join(filtered_sent_list)


def readNews():
    print('reach here')
    sys.stdout.flush()

    clean_stock_news = []
    stock_news = pickle.load(open(news_path, 'rb'))
    i = 0
    for date, news in stock_news:
        print(i)
        i += 1
        sentences = clean_doc(news)
        if len(sentences) != 0:
            clean_stock_news.append([date, sentences])
    clean_stock_news = removeDuplicate(clean_stock_news)

    print(len(clean_stock_news))
    sys.stdout.flush()
    return clean_stock_news


def main():
    clean_news = readNews()
    pickle.dump(clean_news, open(clean_news_path, 'wb'))


if __name__ == '__main__':
    main()
