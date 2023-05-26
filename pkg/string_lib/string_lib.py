import re


STOPWORDS = './pkg/string_lib/vietnamese-stopwords-dash.txt'
with open(STOPWORDS, "r") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n')
        stopwords.append(dd)
    stopwords = set(stopwords)

def santinize_vietnamese_text(text: str) -> str:
    text = filter_stop_words(text, stopwords)
    return remove_emoji(text)

def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split() if word not in stop_words]
    return ' '.join(new_sent)

def remove_emoji(text):
    # Remove emoticons
    text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P|v)', '', text)

    # Remove Unicode emoji characters
    emoji_pattern = re.compile("["
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F910-\U0001F92F"  # more emoticons
        u"\U000024C2-\U0001F251"  # enclosed characters
        u"\U00002702-\U000027B0"  # other miscellaneous symbols
        u"\u2639-\u263A"          # more emoticons
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
