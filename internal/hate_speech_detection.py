import pickle
import numpy as np
import pandas as pd
from pyvi.ViTokenizer import ViTokenizer
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences

from pkg.firebase.firebase import download_file_from_bucket
from pkg.string_lib.string_lib import santinize_vietnamese_text

download_file_from_bucket(
    "./models/vihsd/Text_CNN_model_v13.h5", "models/vihsd/Text_CNN_model_v13.h5",
)
model = keras.models.load_model('./models/vihsd/Text_CNN_model_v13.h5')

with open('./models/vihsd/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

SEQUENCE_LENGTH = 100
labels = ["clean","offensive","hate"]

def does_content_have_hate_speech(text: str) -> bool:
    return bool(hate_speech_detection(text) != 0)

def hate_speech_detection(text: str) -> int:
    pre = pd.Series(preprocess(text))
    text_feat = make_featues(pre)
    prediction = model.predict(text_feat)

    # "clean", "offensive", "hate"
    return prediction.argmax(axis=-1)[0]

def preprocess(text, tokenized = True, lowercased = True) -> str:
    text = ViTokenizer.tokenize(text) if tokenized else text
    text = santinize_vietnamese_text(text)
    return text.lower() if lowercased else text

def pre_process_features(X, y, tokenized = True, lowercased = True):
    X = [preprocess(str(p), tokenized = tokenized, lowercased = lowercased) for p in list(X)]
    for idx, ele in enumerate(X):
        if not ele:
            np.delete(X, idx)
            np.delete(y, idx)
    return X, y

def make_featues(x):
    x = tokenizer.texts_to_sequences(x)
    return pad_sequences(x, maxlen=SEQUENCE_LENGTH)
