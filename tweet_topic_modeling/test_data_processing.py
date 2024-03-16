import pytest
from tweet_topic_modeling.data.utils.data_processing import replace_phrase, process_tweets

@pytest.fixture
def single_tweet():
    return "I have a half pitbull and it's adorable."

@pytest.fixture
def tweet_texts():
    return [
        "I have a half pitbull and it's adorable.",
        "I love my golden retriever."
    ]

@pytest.fixture
def name():
    return "Adila"
    
def test_replace_half_pitbull(single_tweet, name):
    replaced_tweet = replace_phrase(single_tweet, name)
    assert "half pitbull" not in replaced_tweet

def test_replace_golden_retriever(name):
    tweet = "I love my golden retriever."
    replaced_tweet = replace_phrase(tweet, name)
    assert "golden retriever" not in replaced_tweet

def test_replace_with_tom_robbins_phrase(single_tweet, name):
    replaced_tweet = replace_phrase(single_tweet, name)
    print(f'hello: {any(name in replaced_tweet for name in ["Adila", "Patrick", "Sesh"])}')
    assert any(name in replaced_tweet for name in ["Adila", "Patrick", "Sesh"]) is False

def test_process_tweets_returns_list(tweet_texts):
    replaced_tweets = process_tweets(tweet_texts)
    assert isinstance(replaced_tweets, list)

def test_process_tweets_replaces_phrases(tweet_texts):
    replaced_tweets = process_tweets(tweet_texts)
    for tweet, replaced_tweet in zip(tweet_texts, replaced_tweets):
        assert tweet != replaced_tweet

def test_process_tweets_replaces_specific_phrases():
    tweet_texts = ["I have a half pitbull and it's adorable.", "I love my golden retriever."]
    replaced_tweets = process_tweets(tweet_texts)
    assert "half pitbull" not in replaced_tweets[0]
    assert "golden retriever" not in replaced_tweets[1]

def test_replace_with_name():
    tweet = "I love my golden retriever."
    replaced_tweet = replace_phrase(tweet, "TestName")
    assert "TestName" in replaced_tweet