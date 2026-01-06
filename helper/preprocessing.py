import re
import unicodedata
from unidecode import unidecode
from indoNLP.preprocessing import replace_slang, replace_word_elongation, emoji_to_words

def emoji_alias(text):
    text = emoji_to_words(text, delimiter = (" ", " "))
    return " ".join(word.replace("_", " ") for word in text.split())

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    text = replace_slang(text)
    text = replace_word_elongation(text)
    text = emoji_alias(text)
    text = unidecode(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'([^\w\s])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_batch(texts):
    return [clean_text(t) for t in texts]