import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from tqdm import tqdm
import string
import nltk
from nltk.corpus import stopwords
import nltk.stem as stemmer
from nltk.stem.porter import *
from sklearn.model_selection import GridSearchCV
import string
from sklearn.preprocessing import LabelEncoder
# mount your Google Drive, so that you can read data from it.
# Note: it needs your authorization.
# from google.colab import drive
# drive.mount('/content/drive')
# stemming tool from nltk
stemmer = PorterStemmer()
# a mapping dictionary that help remove punctuations
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

# Code Reecieved from Professor for solution Quiz 1
def get_tokens(text):
    # turn document into lowercase
    lowers = text.lower()
    # remove punctuation
    no_punctuation = lowers.translate(remove_punctuation_map)
    # tokenize document
    tokens = nltk.word_tokenize(no_punctuation)
    # stop words
    filtered = [w for w in tokens if not w in stopwords.words("english")]
    # stemming process
    stemmed = []
    for item in filtered:
        stemmed.append(stemmer.stem(item))

    return stemmed

def get_dict(fpath):
    dictionary = {}


    with open(fpath, "r") as f:
        for i, word in enumerate(f):
            dictionary[word.strip()] = i

    return dictionary

def get_doc_tf(word_set, dictionary):
    n_words = len(dictionary)
    tf_vec = np.zeros(n_words)

    max_cnt = 0
    for word in word_set:
        idx = dictionary[word]
        tf_vec[idx] += 1.0

        if tf_vec[idx] > max_cnt:
            max_cnt = tf_vec[idx]

    return tf_vec / max_cnt

def filter_top_k(counter_sorted, limit):
    top_k = {}

    for i, k in enumerate(counter_sorted.keys()):
        if i == limit:
            break
        top_k[k] = counter_sorted[k]

    return top_k

def get_tf_idf(tf_dict, df_vec, n_doc, n_words):

    tf_idf_mtx = np.zeros((n_doc, n_words))
    idf = np.log(n_doc / df_vec)

    for doc_idx, tf_vec in tf_dict.items():
        tf_idf = tf_dict[doc_idx]*idf

        tf_idf_mtx[doc_idx, :] = tf_idf

    return tf_idf_mtx

def tfidf_main(fpath, dictionary):


    n_words = len(dictionary)
    tf = {}
    doc_freq = np.zeros(n_words)

    with open(fpath, 'r') as f:

        lines = f.readlines()
        n_doc = len(lines) - 1

        for i, line in tqdm(enumerate(lines), total=n_doc+1):
            if i == 0:
                continue

            doc_idx = i - 1

            if(len(line.split(",")) == 2):
                id, txt = line.split(",")
            else:
                id, txt, cat = line.split(",")
                cat = cat.strip()
            tokens = get_tokens(txt)

            filtered = []
            filtered_unique = set()
            for word in tokens:
                if word in dictionary:
                    filtered.append(word)
                    filtered_unique.add(word)

            # get term frequency
            tf_vec = get_doc_tf(filtered, dictionary)
            tf[doc_idx] = tf_vec

            # get doc frequency:
            for word in filtered_unique:
                idx = dictionary[word]
                doc_freq[idx] += 1
            
    tfidf_mtx = get_tf_idf(tf, doc_freq, n_doc, n_words) 
    return tfidf_mtx

def tfidf_test(fpath, dictionary):

    n_words = len(dictionary)
    tf = {}
    doc_freq = np.zeros(n_words)

    with open(fpath, 'r') as f:

        lines = f.readlines()
        n_doc = len(lines) - 1

        for i, line in tqdm(enumerate(lines), total=n_doc+1):
            if i == 0:
                continue

            doc_idx = i - 1

            id, txt = line.split(",")
            tokens = get_tokens(txt)

            filtered = []
            filtered_unique = set()
            for word in tokens:
                if word in dictionary:
                    filtered.append(word)
                    filtered_unique.add(word)

            # get term frequency
            tf_vec = get_doc_tf(filtered, dictionary)
            tf[doc_idx] = tf_vec

            # get doc frequency:
            for word in filtered_unique:
                idx = dictionary[word]
                doc_freq[idx] += 1


    tfidf_mtx = get_tf_idf(tf, doc_freq, n_doc, n_words)


    return tfidf_mtx

def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    return lines

dictionary = get_dict("dictionary.txt")
tfidf = tfidf_main("news-train.csv", dictionary)
test_tfidf = tfidf_test("news-test.csv", dictionary)

training = pd.read_csv('news-train.csv')
test = pd.read_csv('news-test.csv')
categories = training['Category']
le = preprocessing.LabelEncoder()
all_y = le.fit_transform(categories)
y_train = le.fit_transform(categories)

tester = RandomForestClassifier(criterion='gini', n_estimators=1000, max_features='log2', min_samples_leaf=1)
tester.fit(tfidf,all_y)
pleasework = tfidf_main("news-test.csv", dictionary)


y_pred = tester.predict(pleasework)
y_pred_label = le.inverse_transform(y_pred)

df_test = pd.read_csv('news-test.csv')

df_test['Category'] = y_pred_label
df_test[['ArticleId', 'Category']].to_csv('labels.csv', header=False, index=False)
