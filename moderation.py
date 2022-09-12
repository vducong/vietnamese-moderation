import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import py_vncorenlp

train = pd.read_csv('./input/train_vn.csv', sep=";")
test = pd.read_csv('../input/test_vn.csv', sep=";")

label_cols = ['toxic']
train['none'] = 1-train[label_cols].max(axis=1)

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

model = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
def tokenize(s):
    # re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    # rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="./vncorenlp")
    output = model.word_segment(s)
    return output

n = train.shape[0]
vec = TfidfVectorizer(
    ngram_range=(1,2), tokenizer=tokenize,
    min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
    smooth_idf=1, sublinear_tf=1,
)

x = vec.fit_transform(train[COMMENT])           # train_term_doc
test_x = vec.transform(test[COMMENT])    # test_term_doc

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
