import re


def lowercase(sentence):
    words = sentence.split(" ")
    words = [word.lower() for word in words]
    return " ".join(words)


def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)
