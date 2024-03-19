import pytest
from gensim.models import LdaModel
from gensim import corpora
import joblib
from models.build_m2_models import create_mapping, preprocess_text, get_coherence_score, main 

def test_create_mapping(unique_tweet_texts, tweet_texts_small):
    mapping = create_mapping(unique_tweet_texts, tweet_texts_small)
    
    assert isinstance(mapping, dict)
    assert len(mapping.keys()) == len(unique_tweet_texts)
    assert all(isinstance(value, list) for value in mapping.values())
    assert all(all(isinstance(text, str) for text in texts) for texts in mapping.values())
    assert [v for v in mapping.keys()] == unique_tweet_texts

def test_preprocess_text():
    text = "This is a test tweet with a URL: https://example.com and a user mention: @user123"
    test_text = "test tweet url user mention"
    preprocessed_text = preprocess_text(text)
    assert preprocessed_text == test_text
    
def test_get_coherence_score(lda_model, train_tokens, dictionary):
    coherence_score_umass = get_coherence_score(lda_model, train_tokens, dictionary, 'u_mass')
    coherence_score_cv = get_coherence_score(lda_model, train_tokens, dictionary, 'c_v')
    assert isinstance(coherence_score_umass, float)
    assert isinstance(coherence_score_cv , float)
    assert coherence_score_umass < 0
    assert 0 < coherence_score_cv < 1

def test_main(tweet_texts_large):
    # Prepare data for testing
    lda_model, train_corpus, dictionary, df_data = main(tweet_texts_large)
    assert isinstance(lda_model, LdaModel)
    assert isinstance(train_corpus, list)
    assert isinstance(dictionary, corpora.Dictionary)
    assert isinstance(df_data, dict)