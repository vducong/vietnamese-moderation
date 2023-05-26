from blingfire import text_to_sentences

from internal.classification import classify, does_content_have_appropriate_sentiment
from internal.hate_speech_detection import does_content_have_hate_speech
from internal.spam_detection import is_spam, spam_predict


BLACKLISTS = []

def moderation(text: str) -> dict:
    sentences = text_to_sentences(text).split('\n')
    for sentence in sentences:
        res = review_sentence_content(sentence)
        if res["is_legitimate"] is False:
            return res

    return {
        "is_legitimate": True,
        "reason": "pass all checks",
    }

def review_sentence_content(sentence: str) -> dict:
    blacklist_check = does_sentence_contain_blacklist(sentence)
    if blacklist_check["is_legitimate"] is False:
        return blacklist_check

    sentiment_check = does_content_have_appropriate_sentiment(sentence)
    if sentiment_check[0] is True:
        return {
            "is_legitimate": True,
            "reason": "pass sentiment check",
        }

    if sentiment_check[1] is True:
        return {
            "is_legitimate": False,
            "reason": "too negative sentiment",
        }

    hate_speech_check = does_content_have_hate_speech(sentence)
    return {
        "is_legitimate": not hate_speech_check,
        "reason": "hate speech detected" if hate_speech_check is True else "pass hate speech check",
    }

def does_sentence_contain_blacklist(text: str) -> dict:
    for blacklist in BLACKLISTS:
        if blacklist.lower() in text.lower():
            return {
                "is_legitimate": False,
                "reason": "contains blacklisted",
            }

    return { "is_legitimate": True }

def comprehensive_checks(text: str) -> list[dict]:
    sentences = text_to_sentences(text).split('\n')
    res = []
    for sentence in sentences:
        res.append({
            "sentence": sentence,
            "does_contain_blacklist": does_sentence_contain_blacklist(sentence)["is_legitimate"],
            "spam_predict": spam_predict(sentence),
            "sentiment_predict": classify(sentence),
            "hate_speech_check":  does_content_have_hate_speech(sentence),
        })
    return res
