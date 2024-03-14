import random
import re
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import warnings
from models.viz.utils import get_replaced_tweets

warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Preprocess text by removing URLs, user mentions, 'RT', punctuation, non-alphabetic characters, stopwords, and perform stemming."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove "RT"
    text = re.sub(r'\bRT\b', '', text)
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization and lowercase
    words = word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Stemming
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def main(tweet_texts, alpha='symmetric'):
    """Perform topic modeling using LDA."""
    # Preprocessing
    processed_texts = [preprocess_text(text) for text in tweet_texts]

    # unique tweets
    unique_tweet_texts = list(set([preprocess_text(text) for text in tweet_texts]))
    
    # Split data into train and test sets
    train_texts, test_texts = train_test_split(processed_texts, test_size=0.35, random_state=42)
    
    # Tokenize again for training LDA
    train_tokens = [text.split() for text in train_texts]
    
    # Create dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(train_tokens)
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_tokens]
    
    # Perform topic modeling using LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                                id2word=dictionary,
                                                num_topics=5,
                                                random_state=42,
                                                passes=20,
                                                per_word_topics=True,
                                                alpha=alpha)
        
    # Label test set using the trained topic model
    test_corpus = [dictionary.doc2bow(text.split()) for text in test_texts]
    
    return unique_tweet_texts, train_texts, train_tokens, test_texts, lda_model, dictionary, test_corpus, train_corpus