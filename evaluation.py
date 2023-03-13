import os
import glob
import re
import jiwer
import speech_recognition as sr
from tqdm import tqdm

import Define


TAG_MAPPING = {
    "google": {  # https://stackoverflow.com/questions/14257598/what-are-language-codes-in-chromes-implementation-of-the-html5-speech-recogniti/14302134#14302134
        "en": "en",
        "zh": "zh",
        "ko": "ko",
        "jp": "ja",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "ru": "ru",
    },
    "whisper": {  # https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
        "en": "en-US",
        "zh": "zh-CN",
        "ko": "ko",
        "jp": "ja",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
        "ru": "ru",
    },
}

r = sr.Recognizer()
def whisper(wav_path, lang: str):
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Whisper
    try:
        res = r.recognize_whisper(audio, model='large', language=TAG_MAPPING["whisper"][lang])
        # res = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, language="ko-KR")
        return res
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Whisper error; {0}".format(e))
    return ""


def google(wav_path, lang):
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google API
    try:
        res = r.recognize_google(audio, key=None, language=TAG_MAPPING["google"][lang])
        return res
    except sr.UnknownValueError:
        print("Google could not understand audio")
    except sr.RequestError as e:
        print("Google error; {0}".format(e))
    return ""


def cer(raw_text, pred_text, remove_whitespace=False):
    raw_text = re.sub(r'[^\w\s]', '', raw_text)
    pred_text = re.sub(r'[^\w\s]', '', pred_text)
    if remove_whitespace:
        raw_text = raw_text.replace(' ', '')
        pred_text = pred_text.replace(' ', '')
    raw_text = raw_text.upper()
    pred_text = pred_text.upper()
    cer = jiwer.cer(raw_text, pred_text)

    return cer
