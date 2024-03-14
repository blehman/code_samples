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
#from nltk.stem import PorterStemmer

def preprocess_text_m1(text):
    """Preprocess text by removing URLs, user mentions, 'RT', punctuation, non-alphabetic characters, stopwords, and perform stemming."""
    stop_words = set(stopwords.words('english'))
    #stemmer = PorterStemmer()
    
    # Remove URLs, user mentions, 'RT', punctuation, non-alphabetic characters
    text = re.sub(r'http\S+|@\w+|\bRT\b|[^a-zA-Z\s]', '', text)
    
    # Tokenization and lowercase
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    #words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)
    
def preprocess_text_m2(text):
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
      
def pickle_file(obj, obj_name):
    """Save object to a pickle file."""
    with open("data/"+obj_name+"_baseline"+"_25passes"+".pkl", 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        
def main(tweet_texts):
    """Main function to perform topic modeling."""
    # Keep only tweets classified as English
    tweet_texts_filtered = filter_lang(tweet_texts)
    pickle_file(tweet_texts_filtered, "tweet_texts_filtered")
        
    # Clean and unique the tweet texts
    tweet_texts_m1 = [preprocess_text_m1(text) for text in tweet_texts_filtered]
    pickle_file(tweet_texts_m1, "unique_tweet_texts_m1")
    unique_tweet_texts_m2 = list(set([preprocess_text_m2(text) for text in tweet_texts_filtered]))
    pickle_file(unique_tweet_texts_m2, "unique_tweet_texts_m2")
    
    # Split data into train and test sets
    train_texts_m1, test_texts_m1 = train_test_split(tweet_texts_m1, test_size=0.35, random_state=42)
    pickle_file(train_texts_m1, "train_texts_m1")
    pickle_file(test_texts_m1, "test_texts_m1")
    
    train_texts_m2, test_texts_m2 = train_test_split(unique_tweet_texts_m2, test_size=0.35, random_state=42)
    pickle_file(train_texts_m2, "train_texts_m2")
    pickle_file(test_texts_m2, "test_texts_m2")
    
    # Tokenize again for training LDA
    train_tokens_m1 = [text.split() for text in train_texts_m1]
    pickle_file(train_tokens_m1, "train_tokens_m1")
    train_tokens_m2 = [text.split() for text in train_texts_m2]
    pickle_file(train_tokens_m2, "train_tokens_m2")
    
    # Create dictionary and corpus for topic modeling
    dictionary_m1 = corpora.Dictionary(train_tokens_m1)
    pickle_file(dictionary_m1, "dictionary_m1")
    dictionary_m2 = corpora.Dictionary(train_tokens_m2)
    pickle_file(dictionary_m2, "dictionary_m2")
    
    train_corpus_m1 = [dictionary_m1.doc2bow(tokens) for tokens in train_tokens_m1]
    pickle_file(train_corpus_m1, "train_corpus_m1")
    train_corpus_m2 = [dictionary_m2.doc2bow(tokens) for tokens in train_tokens_m2]
    pickle_file(train_corpus_m2, "train_corpus_m2")

    df_data = {
        "num_topics": [],
        "method": [],
        "alpha": [],
        #"eta": [],
        "coherence_scores_umass": [],
        "coherence_scores_cv": [],
        "model": []
    }
    
    for num_topics in range(3, 21):
        modeling_params = [
            {"corpus": train_corpus_m1,
             "id2word": dictionary_m1,
             "num_topics": num_topics,
             "random_state": 42,
             "passes": 20,
             "per_word_topics": True,
             "alpha": 'auto'},
            {"corpus": train_corpus_m1,
             "id2word": dictionary_m1,
             "num_topics": num_topics,
             "random_state": 42,
             "passes": 20,
             "per_word_topics": True,
             "alpha": 'symmetric'},
            {"corpus": train_corpus_m2,
             "id2word": dictionary_m2,
             "num_topics": num_topics,
             "random_state": 42,
             "passes": 20,
             "per_word_topics": True,
             "alpha": 'auto'},
            {"corpus": train_corpus_m2,
             "id2word": dictionary_m2,
             "num_topics": num_topics,
             "random_state": 42,
             "passes":20,
             "per_word_topics": True,
             "alpha": 'symmetric'}
        ]
        for i, param in enumerate(modeling_params):
            # Perform topic modeling using LDA
            lda_model = gensim.models.ldamodel.LdaModel(**param)

            # Determine processing method
            method = "m2"
            if i <= 1:
                method = "m1"

            # Get coherence scores
            coherence_scores_umass = get_coherence_score(lda_model, train_tokens_m1 if i <= 1 else train_tokens_m2, param['id2word'], 'u_mass')
            coherence_scores_cv = get_coherence_score(lda_model, train_tokens_m1 if i <= 1 else train_tokens_m2, param['id2word'], 'c_v')

            # Append data to dataframe
            df_data["num_topics"].append(num_topics)
            df_data["method"].append(method)
            df_data["alpha"].append(param["alpha"])
            #df_data["eta"].append(param.get("eta", 'symmetric' if i <= 3 else 'auto'))
            df_data["coherence_scores_umass"].append(coherence_scores_umass)
            df_data["coherence_scores_cv"].append(coherence_scores_cv)
            df_data["model"].append(lda_model)

        if num_topics % 5 == 0:
            pickle_file(df_data, "df_data")

        print(f"---------------------------------------num_topics=={num_topics}---------------------------------------")

    return df_data
    
