import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import py_vncorenlp


label_cols = ['toxic']
COMMENT = 'comment_text'
segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])

def read_train_file():
    train = pd.read_csv('./input/train_vn.csv', sep=";")
    train['none'] = 1-train[label_cols].max(axis=1) # no label comment
    train[COMMENT].fillna("unknown", inplace=True)  # empty comment
    return train
train = read_train_file()

def read_test_file():
    test = pd.read_csv('./input/test_vn.csv', sep=";")
    test[COMMENT].fillna("unknown", inplace=True)   # empty comment
    return test
test = read_test_file()

def predict():
    x, test_x = create_sparse_matrix()

    preds = np.zeros((len(test), len(label_cols)))
    for i, j in enumerate(label_cols):
        print('check if content is:', j)
        m,r = get_model(x, train[j])
        preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

    print(preds)

def create_sparse_matrix():
    vec = TfidfVectorizer(
        ngram_range=(1,2), tokenizer=tokenize,
        min_df=3, max_df=0.9, strip_accents='unicode', 
        use_idf=1, smooth_idf=1, sublinear_tf=1,
    )

    x = vec.fit_transform(train[COMMENT])
    test_x = vec.transform(test[COMMENT])
    return x, test_x

def tokenize(s):
    re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', s)
    sentences = segmenter.word_segment(s)
    return np.concatenate([sen.split() for sen in sentences])  

def get_model(x, y):
    y = y.values

    r = np.log(naive_bayes(x, 1, y) / naive_bayes(x, 0, y))
    x_nb = x.multiply(r)

    m = LogisticRegression(C=4, dual=False, max_iter=200)
    return m.fit(x_nb, y), r

def naive_bayes(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

