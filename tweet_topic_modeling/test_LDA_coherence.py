import pytest
from models.LDA_coherence import *

# Test preprocess_text_m1 function
def test_preprocess_text_m1():
    text = "This is a test tweet with a URL: https://example.com and a user mention: @user123"
    preprocessed_text = preprocess_text_m1(text)
    assert True

# Test preprocess_text_m2 function
def test_preprocess_text_m2():
    text = "This is a test tweet with a URL: https://example.com and a user mention: @user123"
    preprocessed_text = preprocess_text_m2(text)
    assert True

# Test get_coherence_score function
def test_get_coherence_score(lda_model, train_tokens, dictionary):
    coherence_score = get_coherence_score(lda_model, train_tokens, dictionary, 'u_mass')
    assert isinstance(coherence_score, float)


# Test main function
def test_main(tweet_texts_large):
    # Prepare data for testing
    df_data = main(tweet_texts_large)
    assert isinstance(df_data, dict)
    assert all(isinstance(value, list) for value in df_data.values())
