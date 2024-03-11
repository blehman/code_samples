import random
import re
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    """Preprocess text by removing URLs, user mentions, and 'RT'."""
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

def main(tweet_texts):
    """Main function to perform topic modeling."""
    # Clean and unique the tweet texts
    unique_tweet_texts = list(set([preprocess_text(text) for text in tweet_texts]))

    # Split data into train and test sets
    train_texts, test_texts = train_test_split(unique_tweet_texts, test_size=0.2, random_state=42)

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

    # Predict topics for test set
    test_topic_labels = []
    for doc in test_corpus:
        topic_distribution = lda_model.get_document_topics(doc)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        test_topic_labels.append(dominant_topic)

    # Print example of test tweet with its topic label
    for i in range(5):  # Print labels for the first 5 test tweets
        print("Test tweet:", test_texts[i])
        print("Topic label:", test_topic_labels[i])
        print("")

    # Print topics and their meanings
    print("Topics and their meanings:")
    for idx, topic in lda_model.print_topics(-1):
        print("Topic {}: {}".format(idx, topic))
        
    print("""------------------------------------------------\n
initial_tweets:  {initial_tweet_count}
unique tweets:   {unique_tweet_count}
training tweets: {train_set_size}
test tweets:     {test_set_size}\n
total topics:    {total_topics}
------------------------------------------------\n""".format(initial_tweet_count=len(tweet_texts)
                                                             , unique_tweet_count=len(unique_tweet_texts)
                                                             , train_set_size=len(train_texts)
                                                             , test_set_size=len(test_texts)
                                                             , total_topics=len(lda_model.print_topics(-1))
                                                            )
     )
    
    # Print tweets with topic label and probability
    print("Print 5 tweets with its topic label and probability:")
    for i, doc in enumerate(random.sample(test_corpus,5)):
        print("Test tweet:", test_texts[i])
        topic_distribution = lda_model.get_document_topics(doc)
        for topic, prob in topic_distribution:
            print("Topic label:", topic, "Probability:", prob)
        print("")
