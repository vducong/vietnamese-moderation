import numpy as np
from scipy.special import softmax
from py_vncorenlp import VnCoreNLP
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from pkg.firebase.firebase import download_file_from_bucket
from pkg.string_lib.string_lib import santinize_vietnamese_text


download_file_from_bucket(
    "./models/vispam/pytorch_model.bin", "models/vispam/pytorch_model.bin",
)

SAVED = "models/vispam"
tokenizer = AutoTokenizer.from_pretrained(SAVED)
config = AutoConfig.from_pretrained(SAVED)
model = AutoModelForSequenceClassification.from_pretrained(SAVED)

vncorenlp = VnCoreNLP(annotators=["wseg"])

SPAM_SCORE_THRESHOLD = 0.93

def is_spam(text: str) -> bool:
    res = spam_predict(text)
    if res[0][0] == 'LABEL_0':
        return False
    return bool(res[0][1] >= SPAM_SCORE_THRESHOLD)

def spam_predict(text: str) -> list[tuple[str, float]]:
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    res = []
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = scores[ranking[i]]
        res.append([label, np.round(float(score), 4)])
    return res

def preprocess(text: str, tokenized=True, lowercased=True) -> str:
    text = santinize_vietnamese_text(text)
    text = text.lower() if lowercased else text
    return tokenize(text) if tokenized else text

def tokenize(text: str) -> str:
    sentences = vncorenlp.word_segment(text)
    return " ".join(str(item) for item in sentences)

if __name__ == '__name__':
    print(tokenize('mot con vit xoe ra hai cai canh'))
