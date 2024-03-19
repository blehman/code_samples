import pytest
from tweet_topic_modeling.models import build_m2_models

def test_create_mapping():
    unique_tweet_texts = ['preprocessed text 1', 'preprocessed text 2', 'preprocessed text 3']
    tweet_texts = ['original text 1', 'original text 2', 'original text 3', 'original text 1', 'original text 2']
    mapping = build_m2_models.create_mapping(unique_tweet_texts, tweet_texts)
    
    assert isinstance(mapping, dict)
    assert len(mapping) == len(unique_tweet_texts)
    assert all(isinstance(value, list) for value in mapping.values())
    assert all(all(isinstance(text, str) for text in texts) for texts in mapping.values())
