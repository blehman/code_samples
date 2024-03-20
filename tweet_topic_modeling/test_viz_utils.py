import pytest
import joblib
from models.viz.utils import *

# Replace all import statements with pytest fixtures

@pytest.fixture
def replaced_tweets():
    try:
        replaced_tweets = joblib.load("data/whimsy_tweets.joblib")
    except FileNotFoundError:
        raise Exception("Error: File not found.")
    except Exception as e:
        raise Exception(f"{e}")
    return replaced_tweets

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

def test_coherence_score(lda_model, train_tokens, dictionary):
    c_v, u_mass = coherence_score(lda_model, train_tokens, dictionary)
    assert isinstance(c_v, float)
    assert 0<c_v<1
    assert u_mass<0

