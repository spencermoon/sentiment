import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import collections


# Replace contractions
def replace_contraction(text):
    text = contractions.fix(text)
    
    return text

# Remove URLs
def remove_url(text):
    text = re.sub(r'http\S+', '', text)

    return text

# Tokenize tweets
def tokenize_tweet(text):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    text= tknzr.tokenize(text)
    
    return text

# Remove non-ASCII character
def remove_non_ascii(words):
    new_words = []
    for w in words:
        new_word = unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    
    return new_words

# Remove punctuation
def remove_punctuation(words):
    new_words = []
    for w in words:
        new_word = re.sub(r'[^\w\s]', '', w)
        if new_word != '':
            new_words.append(new_word)
    
    return new_words

# Replace numbers
def replace_number(words):
    p = inflect.engine()
    new_words = []
    for w in words:
        if w.isdigit():
            new_word = p.number_to_words(w)
            new_words.append(new_word)
        else:
            new_words.append(w)
    return new_words

# Remove stopwords
def remove_stopwords(words):
    new_words = []
    for w in words:
        if w not in stopwords.words('english'):
            new_words.append(w)
    return new_words

# Stem words
def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for w in words:
        stem = stemmer.stem(w)
        stems.append(stem)
    
    return stems

# Lemmatize verbs
def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in words:
        lemma = lemmatizer.lemmatize(w, pos='v')
        lemmas.append(lemma)
    
    return lemmas

# Combine normalizing functions
def normalize(words):
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_number(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    
    return words

# Create function for text preprocessing
def preprocess(text):
    text = remove_url(text)
    text = replace_contraction(text)
    words = tokenize_tweet(text)
    words = normalize(words)

    return words