import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from pkg.firebase.firebase import download_file_from_bucket
from pkg.string_lib.string_lib import remove_emoji


download_file_from_bucket(
    "./models/cardiffnlp/pytorch_model.bin", "models/cardiffnlp/pytorch_model.bin",
)

SAVED = "models/cardiffnlp"
tokenizer = AutoTokenizer.from_pretrained(SAVED)
config = AutoConfig.from_pretrained(SAVED)
model = AutoModelForSequenceClassification.from_pretrained(SAVED)

sentiments = ["negative", "neutral", "positive"]
TOO_NEGATIVE_THRESHOLD = 0.7
NEUTRAL_AND_NOT_TOO_NEGATIVE_THRESHOLD = 0.5

# is_content_acceptable, is_too_negative
def does_content_have_appropriate_sentiment(text: str) -> tuple[bool, bool]:
    res = classify(text)
    if res[0][0] == sentiments[2]:
        return True, False

    if res[0][0] == sentiments[1] and (
        res[1][0] == sentiments[2] or res[1][1] <= NEUTRAL_AND_NOT_TOO_NEGATIVE_THRESHOLD
    ):
        return True, False

    if res[0][0] == sentiments[0] and res[0][1] >= TOO_NEGATIVE_THRESHOLD:
        return False, True

    return False, False

def classify(text: str) -> list[tuple[str, float]]:
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    res = []
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = scores[ranking[i]]
        res.append([label, np.round(float(score), 4)])
    return res

# Preprocess text (username and link placeholders)
def preprocess(text: str) -> str:
    words = []
    for word in text.split(" "):
        word = '@user' if word.startswith('@') and len(word) > 1 else word
        word = 'http' if word.startswith('http') else word
        words.append(word)
    return remove_emoji(" ".join(words))
