from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from models.LDA_coherence import preprocess_text_m2
from models.viz.utils import filter_lang, get_replaced_tweets
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

def get_corex_model(list_of_str, text_prep, num_topics=10):
    """
    Train a CorEx topic model on a list of strings.

    Args:
    - list_of_str (list): List of strings to train the model on.
    - text_prep (function): Text preprocessing function.
    - num_topics (int): Number of topics to extract (default is 10).

    Returns:
    - topic_model: Trained CorEx topic model.
    - vectorizer: Fitted CountVectorizer used for text preprocessing.
    - corex_topic_words (list): List of lists containing top words for each topic.
    - preprocessed_tweets (list): List of preprocessed tweet texts.
    """
    # Keep only English language tweets 
    tweet_texts_filtered = filter_lang(list_of_str)
    
    # Preprocessing step 
    preprocessed_tweets = list(set([text_prep(text) for text in tweet_texts_filtered]))
    
    # Convert preprocessed text to a document-term matrix
    vectorizer = CountVectorizer(max_features=10000, max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(preprocessed_tweets)
    
    # Train the CorEx model
    topic_model = ct.Corex(n_hidden=num_topics, seed=42)
    topic_model.fit(doc_term_matrix)
    
    # Get and print the topics
    topics = topic_model.get_topics()
    feature_names = vectorizer.get_feature_names_out()  # Get feature names from vectorizer
    corex_topic_words = []
    for i, topic in enumerate(topics):
        words, _, _ = zip(*topic)
        topic_words = [feature_names[idx] for idx in words]  # Convert token IDs to words
        corex_topic_words.append(topic_words)
        print(f"Topic {i+1}: {' '.join(topic_words)}")
        
    return topic_model, vectorizer, corex_topic_words, preprocessed_tweets

def get_calculated_coherence(topic_model, vectorizer, corex_topic_words, preprocessed_tweets):
    """
    Calculate coherence score for a CorEx topic model.

    Args:
    - topic_model: Trained CorEx topic model.
    - vectorizer: Fitted CountVectorizer used for text preprocessing.
    - corex_topic_words (list): List of lists containing top words for each topic.
    - preprocessed_tweets (list): List of preprocessed tweet texts.

    Returns:
    - coherence_scores (list): Coherence scores using both 'c_v' and 'u_mass' methods for the given topic model.
    """
    # Tokenize the text 
    preprocessed_tweets_tokens = [text.split() for text in preprocessed_tweets]
    
    # Creating the term dictionary, where every unique term is assigned an index
    dictionary = corpora.Dictionary(preprocessed_tweets_tokens)
     
    # Creating corpus using dictionary prepared above
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_tweets_tokens]
    
    # Get coherence score
    cm_corex_cv = CoherenceModel(topics=corex_topic_words, texts=preprocessed_tweets_tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
    cm_corex_umass = CoherenceModel(topics=corex_topic_words, texts=preprocessed_tweets_tokens, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    return [cm_corex_cv.get_coherence(), cm_corex_umass.get_coherence()]



def get_corex_models(list_of_str, text_prep, num_topics_range=(5, 20), printer=False):
    """
    Train a range of CorEx topic models on a list of strings for different numbers of topics.

    Args:
    - list_of_str (list): List of strings to train the models on.
    - text_prep (function): Text preprocessing function.
    - num_topics_range (tuple): Range of number of topics to consider (default is (5, 20)).

    Returns:
    - corex_models (list): List of trained CorEx topic models.
    - vectorizers (list): List of fitted CountVectorizers used for text preprocessing.
    - corex_topic_words_list (list): List of lists containing top words for each topic for each model.
    - preprocessed_tweets (list): List of preprocessed tweet texts.
    """
    corex_models = []
    vectorizers = []
    corex_topic_words_list = []
    preprocessed_tweets = []

    for num_topics in range(*num_topics_range):
        # Keep only English language tweets 
        tweet_texts_filtered = filter_lang(list_of_str)
        
        # Preprocessing step 
        preprocessed_tweets_cur = list(set([text_prep(text) for text in tweet_texts_filtered]))
        preprocessed_tweets.append(preprocessed_tweets_cur)
        
        # Convert preprocessed text to a document-term matrix
        vectorizer = CountVectorizer(max_features=10000, max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(preprocessed_tweets_cur)
        vectorizers.append(vectorizer)
        
        # Train the CorEx model
        topic_model = ct.Corex(n_hidden=num_topics, seed=42)
        topic_model.fit(doc_term_matrix)
        corex_models.append(topic_model)
        
        # Get and print the topics
        topics = topic_model.get_topics()
        feature_names = vectorizer.get_feature_names_out()  # Get feature names from vectorizer
        corex_topic_words = []
        for i, topic in enumerate(topics):
            words, _, _ = zip(*topic)
            topic_words = [feature_names[idx] for idx in words]  # Convert token IDs to words
            corex_topic_words.append(topic_words)
            if printer:
                print(f"Topic {i+1}: {' '.join(topic_words)}")
        
        corex_topic_words_list.append(corex_topic_words)
        
    return corex_models, vectorizers, corex_topic_words_list, preprocessed_tweets


def get_coherence_plot_data(corex_models, vectorizers, corex_topic_words_list, preprocessed_tweets, printer=False):
    """
    Calculate coherence scores for a list of CorEx topic models.

    Args:
    - corex_models (list): List of trained CorEx topic models.
    - vectorizers (list): List of fitted CountVectorizers used for text preprocessing.
    - corex_topic_words_list (list): List of lists containing top words for each topic for each model.
    - preprocessed_tweets (list): List of preprocessed tweet texts.

    Returns:
    - coherence_scores (list): List of coherence scores for each CorEx model.
    """
    df_data = {"c_v":[], "u_mass":[], "num_topics":[], "model_name":[]}
    
    for i, (model, vectorizer, topic_words, tweets) in enumerate(zip(corex_models, vectorizers, corex_topic_words_list, preprocessed_tweets)):
        c_v,u_mass = get_calculated_coherence(model, vectorizer, topic_words, tweets)
        df_data["c_v"].append(c_v)
        df_data["u_mass"].append(u_mass)
        df_data["num_topics"].append(model.n_hidden)
        df_data["model_name"].append("m2corex")
        if printer:
            print(f"Coherence scores for model {i+1} with {model.n_hidden} topics -> cv:{c_v}, umass:{u_mass}")
    return df_data
