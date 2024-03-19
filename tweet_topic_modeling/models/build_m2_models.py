import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import random
import re
import gensim
import pickle as pkl
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import CoherenceModel
from models.viz.utils import filter_lang
from collections import defaultdict
from models.viz.utils import get_replaced_tweets

def create_mapping(unique_tweet_texts, tweet_texts):
    """
    Create a mapping from unique tweet texts to tweet texts.
    
    Args:
    - unique_tweet_texts (list): List of unique preprocessed tweet texts.
    - tweet_texts (list): List of original tweet texts.
    
    Returns:
    - mapping (dict): Mapping from unique tweet texts to tweet texts.
    """
    mapping = defaultdict(list)
    
    # Iterate through each original tweet text
    for tweet_text in tweet_texts:
        # Preprocess the tweet text
        preprocessed_text = preprocess_text(tweet_text)
        
        # If the preprocessed text is in the set of unique preprocessed tweet texts
        if preprocessed_text in unique_tweet_texts:
            # Add the original tweet text to the corresponding list in the mapping
            mapping[preprocessed_text].append(tweet_text)
    
    return mapping   
    
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

def get_coherence_score(lda_model, train_tokens, dictionary, coherence_method):
    """Get coherence score for the given LDA model and method."""
    return CoherenceModel(model=lda_model, texts=train_tokens, dictionary=dictionary, coherence=coherence_method).get_coherence()
        
def main(tweet_texts, num_topics=9,method='m2',alpha='symmetric',passes=20,per_word_topic=True,random_state=42):
    """Main function to perform topic modeling."""
    # setup
    # Keep only tweets classified as English
    tweet_texts_filtered = filter_lang(tweet_texts)
        
    # Clean and unique the tweet texts
    unique_tweet_texts = list(set([preprocess_text(text) for text in tweet_texts_filtered]))

    
    # Split data into train and test sets
    train_texts, test_texts = train_test_split(unique_tweet_texts, test_size=0.35, random_state=random_state)

    
    # Tokenize again for training LDA
    train_tokens = [text.split() for text in train_texts]
    
    # Create dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(train_tokens)
    
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_tokens]

    df_data = {
        "num_topics": [num_topics],
        "method": [method],
        "alpha": ['symmetric'],
        "coherence_scores_umass": [],
        "coherence_scores_cv": [],
        "model": []
    }
    param = {
        "corpus": train_corpus,
        "id2word": dictionary,
        "num_topics": num_topics,
        "random_state": random_state,
        "passes":passes,
        "per_word_topics": per_word_topic,
        "alpha": 'symmetric'
    }
    lda_model = gensim.models.ldamodel.LdaModel(**param)

    # Get coherence scores
    coherence_scores_umass = get_coherence_score(lda_model, train_tokens, param['id2word'], 'u_mass')
    coherence_scores_cv = get_coherence_score(lda_model, train_tokens, param['id2word'], 'c_v')

    # Append data to dataframe
    df_data["coherence_scores_umass"].append(coherence_scores_umass)
    df_data["coherence_scores_cv"].append(coherence_scores_cv)
    df_data["model"].append(lda_model)


    print(f"---------------------------------------num_topics=={num_topics}---------------------------------------")

    return lda_model, train_corpus, dictionary, df_data