import re
import torch
import logging
from transformers import BertTokenizer, BertModel
from gensim.corpora import Dictionary
import gensim
import langid
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models import CoherenceModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """Preprocess text by removing URLs, user mentions, and 'RT'."""
    return re.sub(r'http\S+|@\w+|\bRT\b', '', text)

def tokenize_and_embed_tweets(tokenizer, bert_model, tweets):
    """Tokenize and embed tweets using BERT."""
    embeddings = []
    for tweet in tweets:
        encoded_tweet = tokenizer(tweet, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**encoded_tweet)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return embeddings

def train_lda_model(corpus, dictionary):
    """Train an LDA model."""
    logger.info("Training LDA model...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=5,
                                                 random_state=42,
                                                 passes=10,
                                                 per_word_topics=True)
    logger.info("LDA model trained successfully.")
    return lda_model

def visualize_wordclouds(lda_model):
    """Visualize Word Clouds for each topic."""
    logger.info("Generating Word Clouds...")
    topics = lda_model.show_topics(num_topics=-1, formatted=False)
    for topic_id, words in topics:
        word_freq = {word: freq for word, freq in words}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {topic_id}')
        plt.axis('off')
        plt.show()
    logger.info("Word Clouds generated successfully.")

def visualize_pyldavis(lda_model, corpus, dictionary):
    """Visualize pyLDAvis."""
    logger.info("Generating pyLDAvis visualization...")
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)
    logger.info("pyLDAvis visualization generated successfully.")

def compute_coherence(lda_model, corpus, dictionary, texts):
    """Compute Topic Coherence."""
    logger.info("Computing Topic Coherence...")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info(f'Topic Coherence Score: {coherence_lda}')
    logger.info("Topic Coherence computation completed.")

def main(tweet_texts):
    """Main function to perform topic modeling."""
    try:
        # Keep only tweets classified as English
        english_tweet_texts = [tweet for tweet in tweet_texts if langid.classify(tweet)[0] == 'en']
        
        # Preprocess and tokenize texts
        preprocessed_texts = [preprocess_text(text) for text in english_tweet_texts]
        
        # Load BERT tokenizer and model
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name).eval()
        
        # Tokenize and embed tweets using BERT
        bert_embeddings = tokenize_and_embed_tweets(tokenizer, bert_model, preprocessed_texts)
        
        # Convert BERT embeddings to a format suitable for LDA
        flattened_bert_embeddings = [emb.flatten() for emb in bert_embeddings]
        
        # Train LDA model on bag-of-words representations
        lda_dictionary = Dictionary([word_tokenize(text) for text in preprocessed_texts])
        bow_corpus = [lda_dictionary.doc2bow(word_tokenize(text)) for text in preprocessed_texts]
        lda_model = train_lda_model(bow_corpus, lda_dictionary)

        return lda_model, bow_corpus, lda_dictionary, preprocessed_texts
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


        
    
