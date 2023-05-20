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

# mount your Google Drive, so that you can read data from it.
# Note: it needs your authorization.
# from google.colab import drive
# drive.mount('/content/drive')

# stemming tool from nltk
stemmer = PorterStemmer()
# a mapping dictionary that help remove punctuations
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

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

def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    return lines

# real_file = "tech.txt"
# fake_file = "not_tech.txt"

dictionary = get_dict("dictionary.txt")
tfidf = tfidf_main("news-train.csv", dictionary)

# real_data = load_data(real_file)
# fake_data = load_data(fake_file)

# print(real_data[:5])
# print(fake_data[:5])

random.seed(0)
# random.shuffle(real_data)
# random.shuffle(fake_data)

# real_y = np.ones((len(real_data),))
# fake_y = np.zeros((len(fake_data),))

fileReader = pd.read_csv('news-train.csv')
categories = fileReader['Category']
le = preprocessing.LabelEncoder()
all_y = le.fit_transform(categories)
articles = fileReader['Text']

random.shuffle(articles)
##########################
# num_real_train_val = int(len(real_data) * 0.8)
# num_fake_train_val = int(len(fake_data) * 0.8)
num_train_val = int(len(all_y) * 0.8)

from sklearn.model_selection import train_test_split
train_val_X, test_X, train_val_y, test_y = train_test_split(tfidf, all_y, test_size=0.2, random_state=0)
print(train_val_y)

train_val_data = tfidf[:num_train_val]
test_data = tfidf[num_train_val:]

# train_val_data = real_data[:num_real_train_val] + fake_data[:num_fake_train_val]
# test_data = real_data[num_real_train_val:] + fake_data[num_fake_train_val:]

# train_val_y = train_val_data
# test_y = test_data
##########################

print(len(train_val_data), len(train_val_y))
print(len(test_data), len(test_y))


##########################
vectorizer = CountVectorizer()
# train_val_X = vectorizer.fit_transform(train_val_data)
# test_X = vectorizer.transform(test_data)
# train_val_X = all_y[:num_train_val]
# test_X = all_y[num_train_val:]
##########################

print(train_val_X.shape)
print(test_X.shape)


def dtc_parameter_tune(train_val_X, train_val_y):
    depths = [10, 25, 50, 75, 100, 125, 150]
    gini_train_acc_all = []
    gini_val_acc_all = []
    entr_train_acc_all = []
    entr_val_acc_all = []

    kf = KFold(n_splits = 5)
    for depth in depths:
        gini_train_acc = []
        gini_val_acc = []
        entr_train_acc = []
        entr_val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index,:]
            val_X = train_val_X[val_index,:]

            train_y = train_val_y[train_index]
            val_y = train_val_y[val_index]
            
            gini_dtc = tree.DecisionTreeClassifier(criterion="gini", max_depth=depth)
            gini_dtc.fit(train_X, train_y)
            gini_train_acc.append(gini_dtc.score(train_X, train_y))
            gini_val_acc.append(gini_dtc.score(val_X, val_y))
            entr_dtc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
            entr_dtc.fit(train_X, train_y)
            entr_train_acc.append(entr_dtc.score(train_X, train_y))
            entr_val_acc.append(entr_dtc.score(val_X, val_y))
            ##########################

        avg_gini_train_acc = sum(gini_train_acc) / len(gini_train_acc)
        avg_gini_val_acc = sum(gini_val_acc) / len(gini_val_acc)
        avg_entr_train_acc = sum(entr_train_acc) / len(entr_train_acc)
        avg_entr_val_acc = sum(entr_val_acc) / len(entr_val_acc)
        print("Depth: ", depth)
        print("Training accuracy: ", avg_gini_train_acc * 100, "%")
        print("Validation accuracy: ", avg_gini_val_acc * 100, "%")

        gini_train_acc_all.append(avg_gini_train_acc)
        gini_val_acc_all.append(avg_gini_val_acc)
        entr_train_acc_all.append(avg_entr_train_acc)
        entr_val_acc_all.append(avg_entr_val_acc)

    return depths, gini_train_acc_all, gini_val_acc_all, entr_train_acc_all, entr_val_acc_all

depths, gini_train_acc_all, gini_val_acc_all, entr_train_acc_all, entr_val_acc_all  = dtc_parameter_tune(train_val_X, train_val_y)

# plot training/validation curves
# plt.plot(depths, train_acc_all, marker='.', label="Training accuracy")
# plt.plot(depths, val_acc_all, marker='.', label="Validation accuracy")
# plt.xlabel('Depth of tree')
# plt.ylabel('Accuracy')
# plt.legend()



cv = {
    "Validation Error -Gini" : 1 - gini_val_acc_all[np.argmax(gini_val_acc_all)],
    "Training Error - Gini" : 1 - gini_train_acc_all[np.argmax(gini_val_acc_all)],
    "Validation Error - Entropy" : 1 - entr_val_acc_all[np.argmax(gini_val_acc_all)],
    "Training Error - Entropy" : 1 - entr_train_acc_all[np.argmax(entr_val_acc_all)]
}

fig = plt.figure(figsize = (9, 4))
def addlabels(x,y):
    for i in range(len(x)):
        plt. text(i, y[i], y[i], ha = 'center')

x = list(cv.values())
y = list(cv.keys())

plt.bar(y, x, width = 0.4)
plt.xlabel("Error w.r.t Criterion")
plt.ylabel("Error")

# idx = np.arange(2)
# print(idx)
# gini_err = [1 - gini_train_acc_all[np.argmax(gini_val_acc_all)], 1 - gini_val_acc_all[np.argmax(gini_val_acc_all)]]
# entr_err = [1 - entr_train_acc_all[np.argmax(entr_val_acc_all)], 1 - entr_val_acc_all[np.argmax(entr_val_acc_all)]]
# # df = pd.DataFrame(gini_err, columns=["gini"])
# # df.plot.bar()
# plt.bar(idx, gini_err, 0.4, label='Gini')
# plt.bar(idx + 0.4, entr_err, 0.4, label='Entropy')
# plt.ylabel('Error')
# plt.xticks(idx + 0.4 / 2, ('Train error', 'Val error'))
# plt.legend(loc='best')
# plt.show()



##########################
best_depth = depths[np.argmax(gini_val_acc_all)]
dtc = tree.DecisionTreeClassifier(max_depth=best_depth)
dtc.fit(train_val_X, train_val_y)
train_acc = dtc.score(train_val_X, train_val_y)
test_acc = dtc.score(test_X, test_y)
##########################
print("=========================================")
print("Best depth: ", best_depth)
print("Training accuracy: ", train_acc * 100, "%")
print("Test accuracy: ", test_acc * 100, "%")




p_dtc = {
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': range(1, 20), 
    'max_features': range(200, 1000, 200)
}
CV_DTC = GridSearchCV(tree.DecisionTreeClassifier(), p_dtc, return_train_score=True)
CV_DTC.fit(tfidf, all_y)

# scores = [x[1] for x in CV_DTC.cv_results_]
# scores = np.array(scores).reshape(len(p_dtc['min_samples_leaf']), len(p_dtc['max_features']))

# for ind, i in enumerate('min_samples_leaf'):
#     plt.plot(p_dtc['max_features'], scores[ind], label='criterion' + str(i))
# plt.legend()
# plt.xlabel('Gamma')
# plt.ylabel('Mean score')
# plt.show()

res = pd.DataFrame(CV_DTC.cv_results_['params'])
res['mean_train_score'] = pd.DataFrame(CV_DTC.cv_results_)['mean_train_score']
res['mean_test_score'] = pd.DataFrame(CV_DTC.cv_results_)['mean_test_score']


for i in p_dtc['criterion']:
    plt.figure(figsize=(8, 6))
    for x in p_dtc['max_features']:
        xy = res[(res['criterion'] == i) & (res['max_features'] == x)]
        plt.plot(xy['min_samples_leaf'], 1-xy['mean_train_score'], label=f'{x} (train)')
        plt.plot(xy['min_samples_leaf'], 1-xy['mean_test_score'], linestyle='--', label=f'{x} (valid)')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('Avg. error')
    plt.show()




# from sklearn.ensemble import RandomForestClassifier
# p_rf = {'n_estimators': range(50, 1001, 50)}
# CV_DTC = GridSearchCV(RandomForestClassifier(), p_rf, return_train_score=True)
# CV_DTC.fit(tfidf, all_y)

# # Graphing Average Accuracy
# plt.plot(p_rf['n_estimators'], CV_DTC.cv_results_['mean_test_score'], marker='.', label='val accuracy')
# plt.ylabel('avg accuracy')
# plt.xlabel('number of estimators')
# plt.legend()
# plt.show()
# # Graphing Stadard Deviation Accuracy
# plt.plot(p_rf['n_estimators'], CV_DTC.cv_results_['std_test_score'], marker='.', label='val accuracy standard deviation')
# plt.ylabel('accuracy standard deviation')
# plt.xlabel('number of estimators')
# plt.legend()
# plt.show()



import xgboost as xgb
xgb.set_config(verbosity=0)

xgb_matrix = xgb.DMatrix(np.array(tfidf), label=all_y)
y_list = []
lr = []
for i in range(10, 95, 5):
    params = { 'eta': i/100, 'num_class': 5 }
    xgb_cv = xgb.cv(params, xgb_matrix, num_boost_round=20 )
    y_loc = xgb_cv.iloc[-1].to_dict()
    y_loc['eta'] = i/100
    y_list.append(y_loc)
    lr.append(i/100)

jeffry = pd.DataFrame(y_list)
plt.plot(lr, 1-jeffry['test-mlogloss-mean'], marker='.', label='validation accuracy')
plt.xlabel('eta')
plt.ylabel('Avg. accuracy')
plt.legend()
plt.savefig('4_d1.png')
plt.show()

plt.plot(lr, jeffry['test-mlogloss-std'], marker='.', label='validation accuracy std')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy std')
plt.legend()
plt.savefig('4_d2.png')
plt.show()




