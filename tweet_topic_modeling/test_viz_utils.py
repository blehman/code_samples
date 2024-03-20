import pytest
import joblib
from models.viz.utils import *

# Replace all import statements with pytest fixtures

@pytest.fixture
def replaced_tweets():
    try:
        replaced_tweets = joblib.load("tweet_topic_modeling/data/whimsy_tweets.joblib")
    except FileNotFoundError:
        raise Exception("Error: File not found.")
    except Exception as e:
        raise Exception(f"{e}")
    return replaced_tweets

@pytest.fixture
def tweet_texts():
    return [
        "I have a half pitbull and it's adorable.",
        "I love my golden retriever."
    ]

@pytest.fixture
def test_lda_model():
    # You would need to provide a mock or a test-specific LDA model for this fixture
    return mock_lda_model

@pytest.fixture
def test_corpus():
    # You would need to provide a mock or a test-specific corpus for this fixture
    return mock_corpus

@pytest.fixture
def test_dictionary():
    # You would need to provide a mock or a test-specific dictionary for this fixture
    return mock_dictionary

# Define tests using the fixtures

def test_get_replaced_tweets_exists(replaced_tweets):
    assert replaced_tweets is not None

def test_filter_lang():
    portuguese = "Levei meu pitbull para a praia para nadar com a namorada dele, que é uma golden retriever chamada moo moo"
    german = "Ich nahm meinen Pitbull mit an den Strand, um mit seiner Freundin zu schwimmen, die ein Golden Retriever namens Moo Moo ist"
    english = "I took my pitbull to the beach to swim with his girlriend who is a golden retriever named moo moo"
    lang_tweets = [portuguese, german, english]
    replaced_tweets = filter_lang(lang_tweets)
    assert len(replaced_tweets)==1
    assert english in replaced_tweets

def test_coherence_score(lda_model, corpus, dictionary):
    c_v, u_mass = coherence_score(lda_model, train_tokens, dictionary)
    assert isinstance(c_v, float)
    assert 0<c_v<1
    assert u_mass<0
    
def test_print_results():
    # Write tests for print_results function
    pass

def test_topic_labels():
    # Write tests for topic_labels function
    pass

def test_generate_topic_examples():
    # Write tests for generate_topic_examples function
    pass

def test_generate_word_clouds():
    # Write tests for generate_word_clouds function
    pass

def test_plot_bar_plots():
    # Write tests for plot_bar_plots function
    pass

def test_visualize_pyldavis():
    # Write tests for visualize_pyldavis function
    pass

def test_get_altair_css():
    # Write tests for get_altair_css function
    pass

def test_add_properties():
    # Write tests for add_properties function
    pass

def test_build_fugly_brokeness():
    # Write tests for build_fugly_brokeness function
    pass

def test_add_multiline_properties():
    # Write tests for add_multiline_properties function
    pass

def test_build_multiline_altair():
    # Write tests for build_multiline_altair function
    pass