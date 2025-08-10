import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOP = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_series(series: pd.Series):
    return series.fillna("").map(clean_text).map(tokenize_and_lemmatize)
