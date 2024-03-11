import random
import re
import gensim
import langid
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from models.viz.utils import generate_word_clouds, plot_bar_plots

def preprocess_text(text):
    # Preprocessing steps
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



def main(tweet_texts, print_results=False, word_cloud=False, bar_plots=False, pyLDAvis=False):
    """Main function to perform topic modeling."""
    # Keep only tweets classified as english
    english_tweet_texts = []
    for tweet in tweet_texts:
        lang, _ = langid.classify(tweet)
        if lang == 'en':
            english_tweet_texts.append(tweet)

    # Clean and unique the tweet texts
    unique_english_tweet_texts = list(set([preprocess_text(text) for text in english_tweet_texts]))

    # Split data into train and test sets
    train_texts, test_texts = train_test_split(unique_english_tweet_texts, test_size=0.2, random_state=42)

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
                                                passes=10,
                                                per_word_topics=True)

    # Label test set using the trained topic model
    test_corpus = [dictionary.doc2bow(text.split()) for text in test_texts]

return lda_model, train_corpus, dictionary

    if word_cloud:
        generate_word_clouds(lda_model)
    if bar_plots:
        plot_bar_plots(lda_model)
    if pyLDAvis:
        return lda_model, train_corpus, dictionary

        
    
