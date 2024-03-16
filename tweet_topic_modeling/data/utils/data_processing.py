import random
import re

def generate_phrase(nouns, phrases, adjectives):
    """
    Randomly select items from lists and combine the selections into a str.

    Parameters:
        nouns (list): list of str
        phrases (list): list of str
        adjectives (list): list of str

    Returns:
        str: phrase combining the selected noun, phrase and adjective.
    """
    noun = random.choice(nouns)
    phrase = random.choice(phrases)
    adjective = random.choice(adjectives)
    return f"{adjective} {noun} {phrase}"

def generate_name():
    """
    Generate a random name from the given list.

    Returns:
        str: A random name.
    """
    return random.choice(['Adila', 'Patrick', 'Sesh'])

def generate_wild_adjective(name):
    """
    Randomly select a wild adjective from a unique list keyed by name.

    Parameters:
        name (str): key for dict

    Returns:
        str: adjective.
    """
    wild_adj = {
        'Adila': ['gallimaufrous', 'brouhahic', 'lollygagging', 'nudiustertian', 'hobbledehoyish', 'hullaballooing'],
        'Patrick': ['xertzian', 'wabbitish', 'smellfungous', 'cattywampous', 'smellfungous', 'doozyish'],
        'Sesh': ['ratoonal', 'jiggery-pokeryish', 'zaftigish', 'bumfuzzling', 'hornswoggling', 'donnybrookian']
    }
    return random.choice(wild_adj[name])

def replace_phrase(tweet, name):
    """
    Replace phrases in the tweet.

    Parameters:
        tweet (str): The tweet text.
        name (str): The name to replace with.

    Returns:
        str: The tweet with replaced phrases.
    """
    # Replace "pitbull" with a Tom Robbins-style phrase
    tom_robbins_phrase = generate_phrase(nouns=["pitbull"], phrases=["half"], adjectives=["friendly"])
    replaced_tweet = re.sub(r'half\s+pitbull', f"half {tom_robbins_phrase},", tweet, flags=re.IGNORECASE)

    # Replace "golden retriever" with the given name
    replaced_tweet = re.sub(r'golden\s+retriever', name, replaced_tweet, flags=re.IGNORECASE)

    return replaced_tweet

def process_tweets(tweet_texts):
    """
    Process tweets by replacing phrases.

    Parameters:
        tweet_texts (list): List of tweet texts.

    Returns:
        list: List of processed tweets.
    """
    replaced_tweets = []
    for tweet in tweet_texts:
        name = generate_name()
        replaced_tweets.append(replace_phrase(tweet, name))
    return replaced_tweets