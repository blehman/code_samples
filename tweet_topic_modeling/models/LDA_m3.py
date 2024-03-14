import random
import re
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import CoherenceModel
from models.viz.utils import filter_lang


def preprocess_text(text):
    """Preprocess text by removing URLs, user mentions, and 'RT'."""
    
    stop_words = set(stopwords.words('english'))

    # Remove URLs, user mentions, 'RT', punctuation, non-alphabetic characters
    text = re.sub(r'http\S+|@\w+|\bRT\b|[^a-zA-Z\s]', '', text)
    
    # Tokenization, lowercase, and removal of stopwords
    words = word_tokenize(text.lower())
    stop_words = stop_words.union(set(['half']))
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def main(tweet_texts, alpha='auto', num_topics=5):
    """Main function to perform topic modeling."""
    # Keep only tweets classified as english
    tweet_texts_filtered = filter_lang(tweet_texts)
    
    # Clean and unique the tweet texts
    unique_tweet_texts = list(set([preprocess_text(text) for text in tweet_texts_filtered]))

    #processed_texts = [preprocess_text(text) for text in tweet_texts]
    # Split data into train and test sets | note: tweet_texts could be exchanged with unique_tweet_texts
    train_texts, test_texts = train_test_split(unique_tweet_texts, test_size=0.35, random_state=42)

    # Tokenize again for training LDA
    train_tokens = [text.split() for text in train_texts]

    # Create dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(train_tokens)
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_tokens]

    # Perform topic modeling using LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=42,
                                                passes=20,
                                                per_word_topics=True,
                                                alpha='auto')
    
    # Label test set using the trained topic model
    test_corpus = [dictionary.doc2bow(text.split()) for text in test_texts]

    return lda_model, tweet_texts, num_toipcs, unique_tweet_texts, train_texts, train_tokens, test_texts, dictionary, test_corpus, train_corpus




        
    
