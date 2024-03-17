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
import joblib
import datetime 
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
      
        
def main(tweet_texts):
    """Main function to perform topic modeling."""
    # setup
    time_created = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    passes = 20
    random_state = 42
    per_word_topics = True
    model_file_path = "data/lda/"
    # Keep only tweets classified as English
    tweet_texts_filtered = filter_lang(tweet_texts)
    joblib.dump(tweet_texts_filtered, f"{model_file_path}tweet_texts_filtered_{time_created}.joblib")
        
    # Clean and unique the tweet texts
    tweet_texts_m1 = [preprocess_text_m1(text) for text in tweet_texts_filtered]
    joblib.dump(tweet_texts_m1, f"{model_file_path}unique_tweet_texts_m1_{time_created}.joblib")
    unique_tweet_texts_m2 = list(set([preprocess_text_m2(text) for text in tweet_texts_filtered]))
    joblib.dump(unique_tweet_texts_m2, f"{model_file_path}unique_tweet_texts_m2_{time_created}.joblib")
    
    # Split data into train and test sets
    train_texts_m1, test_texts_m1 = train_test_split(tweet_texts_m1, test_size=0.35, random_state=42)
    joblib.dump(train_texts_m1, f"{model_file_path}train_texts_m1_{time_created}.joblib")
    joblib.dump(test_texts_m1, f"{model_file_path}test_texts_m1_{time_created}.joblib")
    
    train_texts_m2, test_texts_m2 = train_test_split(unique_tweet_texts_m2, test_size=0.35, random_state=42)
    joblib.dump(train_texts_m2, f"{model_file_path}train_texts_m2_{time_created}.joblib")
    joblib.dump(test_texts_m2, f"{model_file_path}test_texts_m2_{time_created}.joblib")
    
    # Tokenize again for training LDA
    train_tokens_m1 = [text.split() for text in train_texts_m1]
    joblib.dump(train_tokens_m1, f"{model_file_path}train_tokens_m1_{time_created}.joblib")
    train_tokens_m2 = [text.split() for text in train_texts_m2]
    joblib.dump(train_tokens_m2, f"{model_file_path}train_tokens_m2_{time_created}.joblib")
    
    # Create dictionary and corpus for topic modeling
    dictionary_m1 = corpora.Dictionary(train_tokens_m1)
    joblib.dump(dictionary_m1, f"{model_file_path}dictionary_m1_{time_created}.joblib")
    dictionary_m2 = corpora.Dictionary(train_tokens_m2)
    joblib.dump(dictionary_m2, f"{model_file_path}dictionary_m2_{time_created}.joblib")
    
    train_corpus_m1 = [dictionary_m1.doc2bow(tokens) for tokens in train_tokens_m1]
    joblib.dump(train_corpus_m1, f"{model_file_path}train_corpus_m1_{time_created}.joblib")
    train_corpus_m2 = [dictionary_m2.doc2bow(tokens) for tokens in train_tokens_m2]
    joblib.dump(train_corpus_m2, f"{model_file_path}train_corpus_m2_{time_created}.joblib")

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
             "random_state": random_state,
             "passes": passes,
             "per_word_topics": per_word_topics,
             "alpha": 'auto'},
            {"corpus": train_corpus_m1,
             "id2word": dictionary_m1,
             "num_topics": num_topics,
             "random_state": random_state,
             "passes": passes,
             "per_word_topics": per_word_topics,
             "alpha": 'symmetric'},
            {"corpus": train_corpus_m2,
             "id2word": dictionary_m2,
             "num_topics": num_topics,
             "random_state": random_state,
             "passes": passes,
             "per_word_topics": per_word_topics,
             "alpha": 'auto'},
            {"corpus": train_corpus_m2,
             "id2word": dictionary_m2,
             "num_topics": num_topics,
             "random_state": random_state,
             "passes":passes,
             "per_word_topics": per_word_topics,
             "alpha": 'symmetric'}
        ]
        for i, param in enumerate(modeling_params):
            # Determine processing method
            method = "m2"
            if i <= 1:
                method = "m1"
                
            # Perform topic modeling using LDA
            lda_model = gensim.models.ldamodel.LdaModel(**param)
            lda_model.save(f"{model_file_path}lda_{method+param['alpha']+str(num_topics)}.model")
            
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

        print(f"---------------------------------------num_topics=={num_topics}---------------------------------------")

    return df_data
    
