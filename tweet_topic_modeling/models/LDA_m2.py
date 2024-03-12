import random
import re
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import CoherenceModel


def preprocess_text(text):
    """Preprocess text by removing URLs, user mentions, and 'RT'."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Remove URLs, user mentions, 'RT', punctuation, non-alphabetic characters
    text = re.sub(r'http\S+|@\w+|\bRT\b|[^a-zA-Z\s]', '', text)
    
    # Tokenization, lowercase, and removal of stopwords
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    #words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def main(tweet_texts):
    """Main function to perform topic modeling."""
    # Clean and unique the tweet texts
    unique_tweet_texts = list(set([preprocess_text(text) for text in tweet_texts]))
    clean_tweet_texts = [preprocess_text(text) for text in tweet_texts]
    # Split data into train and test sets | note: tweet_texts could be exchanged with unique_tweet_texts
    train_texts, test_texts = train_test_split(clean_tweet_texts, test_size=0.2, random_state=42)

    # Tokenize again for training LDA
    train_tokens = [text.split() for text in train_texts]

    # Create dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(train_tokens)
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_tokens]

    # Perform topic modeling using LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                                id2word=dictionary,
                                                num_topics=4,
                                                random_state=42,
                                                passes=10,
                                                per_word_topics=True)

    # Label test set using the trained topic model
    test_corpus = [dictionary.doc2bow(text.split()) for text in test_texts]

    return tweet_texts, unique_tweet_texts, train_texts, test_texts, lda_model, dictionary, test_corpus
    # Call the print_results function from the result printer module
    