import random
import re
import pickle as pkl

def generate_phrase(nouns,phrases,adjectives):
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
    
def generate_adila_phrase():
    """
    Generate a Tom Robbins-style phrase for Adila.
    """
    adila_weird_nouns = ["gazelle", "volcano", "parrot", "bouncehouse", "mermaid", "cloud", "sunflower", "trampoline"]
    adila_ridiculous_phrases = ["dancing with fireflies", "surfing on rainbows", "moonwalking on the ocean", "disco dancing with penguins", "cartwheeling through galaxies"]
    adila_absurd_adjectives = ["sparkling", "celestial", "whimsical", "effervescent", "ethereal", "luminous", "resplendent", "radiant"]
    
    return generate_phrase(adila_weird_nouns,adila_ridiculous_phrases,adila_absurd_adjectives)

def generate_patrick_phrase():
    """
    Generate a Tom Robbins-style phrase for Patrick.
    """
    patrick_weird_nouns = ["sphinx", "whirlwind", "tornado", "centaur", "meteor", "avalanche", "chimera", "typhoon"]
    patrick_ridiculous_phrases = ["surfing on sand dunes", "dancing with ghosts", "whispering to trees", "juggling thunderbolts", "swimming in a sea of stars"]
    patrick_absurd_adjectives = ["thundering", "cosmic", "mystical", "enigmatic", "surreal", "frenetic", "vibrant", "captivating"]

    return generate_phrase(patrick_weird_nouns,patrick_ridiculous_phrases,patrick_absurd_adjectives)
    
def generate_sesh_phrase():
    """
    Generate a Tom Robbins-style phrase for Sesh.
    """
    sesh_weird_nouns = ["gargoyle", "moonbeam", "whirlpool", "quasar", "lightning", "shooting star", "volcano", "solar flare"]
    sesh_ridiculous_phrases = ["riding on rainbow unicorns", "sailing on a sea of dreams", "whispering to the moon", "dancing with auroras", "juggling comets"]
    sesh_absurd_adjectives = ["splendiferous", "peculiar", "transcendent", "enchanting", "mythical", "fantastical", "spellbinding", "magnificent"]
    
    return generate_phrase(sesh_weird_nouns,sesh_ridiculous_phrases,sesh_absurd_adjectives)

def generate_wild_adjective(name):
    """
    note: being considered for deletion. 
    Randomly select a wild adjective from a unique list keyed by name.

    Parameters:
        name (str): key for dict

    Returns:
        str: adjective.
    """
    wild_adj = {
        'Adila': ['gallimaufrous', 'brouhahic', 'lollygagging', 'nudiustertian', 'hobbledehoyish', 'hullaballooing'],
        'Patrick': ['xertzian', 'wabbitish', 'smellfungous', 'cattywampous', 'smellfungous','doozyish'],
        'Sesh':['ratoonal', 'jiggery-pokeryish', 'zaftigish', 'bumfuzzling', 'hornswoggling', 'donnybrookian']
                }
    return random.choice(wild_adj[name])

def replace_phrase(tweet):
    """
    Replace phrases in the tweet.

    Parameters:
        tweet (str): The tweet text.

    Returns:
        str: The tweet with replaced phrases.
    """
    name = random.choice(['Adila', 'Patrick', 'Sesh'])
    if name == 'Adila':
        tom_robbins_phrase = generate_adila_phrase()
    elif name == 'Patrick':
        tom_robbins_phrase = generate_patrick_phrase()
    else:
        tom_robbins_phrase = generate_sesh_phrase()
    
    # Replace "golden retriever" with An Emily Dickinson adjective and one of the names 'Adila', 'Patrick', or 'Sesh'
    replaced_tweet = re.sub(r'golden\s+retriever', f'{name}', tweet, flags=re.IGNORECASE) #{generate_wild_adjective(name)} 
    
    # Replace "pitbull" with a Tom Robbins-style phrase
    replaced_tweet = re.sub(r'half\s+pitbull', f"half {tom_robbins_phrase}", replaced_tweet, flags=re.IGNORECASE)
    
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
        replaced_tweets.append(replace_phrase(tweet))
    return replaced_tweets

if __name__ == "__main__":
    tweet_texts = tweet_text  # List of 5000 tweets
    replaced_tweets = process_tweets(tweet_texts)