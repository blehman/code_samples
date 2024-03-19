import pytest
from data.utils.data_processing import *
    
def test_replace_half_pitbull(single_tweet):
    replaced_tweet = replace_phrase(single_tweet)
    assert "half pitbull" not in replaced_tweet

def test_replace_golden_retriever():
    tweet = "I love my golden retriever."
    replaced_tweet = replace_phrase(tweet)
    assert "golden retriever" not in replaced_tweet

def test_replace_with_tom_robbins_phrase(tweet_texts, names):
    replaced_tweet = replace_phrase(tweet_texts[0])
    assert any([name in replaced_tweet for name in names]) is False

def test_process_tweets_returns_list(tweet_texts):
    replaced_tweets = process_tweets(tweet_texts)
    assert isinstance(replaced_tweets, list)

def test_process_tweets_replaces_phrases(tweet_texts):
    replaced_tweets = process_tweets(tweet_texts)
    for tweet, replaced_tweet in zip(tweet_texts, replaced_tweets):
        assert tweet != replaced_tweet

def test_process_tweets_replaces_specific_phrases(tweet_texts):
    replaced_tweets = process_tweets(tweet_texts)
    assert "half pitbull" not in replaced_tweets[0]
    assert "golden retriever" not in replaced_tweets[1]

def test_replace_with_name(single_tweet, names):
    replaced_tweet = replace_phrase(single_tweet)
    assert any([name in replaced_tweet for name in names])
    
# Test generate_phrase function
def test_generate_phrase():
    nouns = ['noun1', 'noun2', 'noun3']
    phrases = ['phrase1', 'phrase2', 'phrase3']
    adjectives = ['adjective1', 'adjective2', 'adjective3']
    phrase = generate_phrase(nouns, phrases, adjectives)
    assert isinstance(phrase, str)
    assert any(noun in phrase for noun in nouns)
    assert any(phrase in phrase for phrase in phrases)
    assert any(adjective in phrase for adjective in adjectives)

# Test generate_adila_phrase function
def test_generate_adila_phrase():
    phrase = generate_adila_phrase()
    assert isinstance(phrase, str)
    assert any(noun in phrase for noun in ['gazelle', 'volcano', 'parrot', 'bouncehouse', 'mermaid', 'cloud', 'sunflower', 'trampoline'])

# Test generate_patrick_phrase function
def test_generate_patrick_phrase():
    phrase = generate_patrick_phrase()
    assert isinstance(phrase, str)
    assert any(noun in phrase for noun in ['sphinx', 'whirlwind', 'tornado', 'centaur', 'meteor', 'avalanche', 'chimera', 'typhoon'])

# Test generate_sesh_phrase function
def test_generate_sesh_phrase():
    phrase = generate_sesh_phrase()
    assert isinstance(phrase, str)
    assert any(noun in phrase for noun in ['gargoyle', 'moonbeam', 'whirlpool', 'quasar', 'lightning', 'shooting star', 'volcano', 'solar flare'])

# Test generate_wild_adjective function
def test_generate_wild_adjective():
    adj = generate_wild_adjective('Adila')
    assert isinstance(adj, str)
    assert adj in ['gallimaufrous', 'brouhahic', 'lollygagging', 'nudiustertian', 'hobbledehoyish', 'hullaballooing']

# Test replace_phrase function
def test_replace_phrase():
    tweet = "This is a tweet with a half pitbull and a golden retriever."
    replaced_tweet = replace_phrase(tweet)
    assert isinstance(replaced_tweet, str)
    assert 'half pitbull' not in replaced_tweet
    assert 'golden retriever' not in replaced_tweet

# Test process_tweets function
def test_process_tweets():
    tweets = ["This is a tweet with a half pitbull and a golden retriever.",
              "Another tweet with a half pitbull and a golden retriever."]
    processed_tweets = process_tweets(tweets)
    assert isinstance(processed_tweets, list)
    assert all(isinstance(tweet, str) for tweet in processed_tweets)
    assert all('half pitbull' not in tweet for tweet in processed_tweets)
    assert all('golden retriever' not in tweet for tweet in processed_tweets)
