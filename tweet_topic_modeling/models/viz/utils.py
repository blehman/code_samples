import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models import CoherenceModel
import pickle as pkl
import random
import langid
import altair as alt
import numpy as np
import pandas as pd
from IPython.display import display
import altair as alt

plt.rcParams["figure.figsize"] = [10,5]

def get_replaced_tweets():
    if 'replaced_tweets' not in locals() and 'replaced_tweets' not in globals(): 
        try:
            with open("data/whimsy_tweets.pkl", 'rb') as f:
                replaced_tweets = pkl.load(f)
        except FileNotFoundError:
            print("Error: File not found.")
            return None
        except Exception as e:
            print("Error:", e)
            return None
    return replaced_tweets
    
def filter_lang(tweet_texts):
    # Keep only tweets classified as english
    english_tweet_texts = []
    for tweet in tweet_texts:
        lang, _ = langid.classify(tweet)
        if lang == 'en':
            english_tweet_texts.append(tweet)
    return english_tweet_texts

def coherence_score(lda_model, train_tokens, dictionary):
    coherence_model_lda = CoherenceModel(model=lda_model, texts=train_tokens, dictionary=dictionary, coherence='c_v')
    c_v = coherence_model_lda.get_coherence()
    coherence_model_lda = CoherenceModel(model=lda_model, texts=train_tokens, dictionary=dictionary, coherence='u_mass')
    u_mass=coherence_model_lda.get_coherence()
    return c_v, u_mass
    
def print_results(tweet_texts, unique_tweet_texts, train_texts, test_texts, lda_model, train_tokens, dictionary, test_corpus):
    """Print the results of topic modeling."""
    # Print statistics
    print(f"Initial tweets: {len(tweet_texts)}")
    print(f"Unique tweets: {len(unique_tweet_texts)}")
    print(f"Training tweets: {len(train_texts)}")
    print(f"Test tweets: {len(test_texts)}")
    print(f"Total topics: {len(lda_model.print_topics(-1))}\n")

    # Print topics and their meanings
    print("Topics and their meanings:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")
        
    print(f"\nCoherence Score: {coherence_score(lda_model, train_tokens, dictionary)}\n")

    # Print tweets with topic label and probability
    print("Print 5 tweets with their topic label and probability:")
    for i, doc in enumerate(random.sample(test_corpus, 5)):
        print("Test tweet:", test_text[i])
        topic_distribution = lda_model.get_document_topics(doc)
        for topic, prob in topic_distribution:
            print("Topic label:", topic, "Probability:", prob)
        print("")

def topic_labels(lda_model, test_corpus):
    # Predict topics for test set
    test_topic_labels = []
    for doc in test_corpus:
        topic_distribution = lda_model.get_document_topics(doc)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        test_topic_labels.append(dominant_topic)
        
def generate_topic_examples(lda_model, test_corpus, test_texts):
    # Predict topics for test set
    test_topic_labels = []
    for doc in test_corpus:
        topic_distribution = lda_model.get_document_topics(doc)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        test_topic_labels.append(dominant_topic)

    # Print example of test tweet with its topic label
    for i in range(0, lda_model.num_topics-1):  # Print labels for the first 5 test tweets
        print("Test tweet:", test_texts[i])
        print("Topic label:", test_topic_labels[i])
        print("")

    # Print topics and their meanings
    print("Topics and their meanings:")
    for idx, topic in lda_model.print_topics(-1):
        print("Topic {}: {}".format(idx, topic))

def generate_word_clouds(lda_model):
    # Generate word clouds for each topic
    for idx, topic in lda_model.show_topics(formatted=False):
        word_freq = {word: freq for word, freq in topic}
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_freq)
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Topic {}'.format(idx))
        plt.axis('off')
        plt.show()

def plot_bar_plots(lda_model):
    # Create bar plots for the most frequent terms in each topic
    for idx, topic in lda_model.show_topics(formatted=False):
        terms, freqs = zip(*topic)
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(terms)), freqs, align='center', color='skyblue')
        plt.yticks(range(len(terms)), terms)
        plt.gca().invert_yaxis()
        plt.xlabel('Frequency')
        plt.title('Topic {}'.format(idx))
        plt.show()
        
def visualize_pyldavis(lda_model, corpus, dictionary):
    """Visualize pyLDAvis."""
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)



def get_altair_css():
    return {
    'configure_title':{
        'fontSize':30
        , 'font':'Courier'
        , 'anchor': 'start'
        , 'color': 'gray'
    },
    'properties':{
        'width':800,
        'height':400
    },
    'configure_legend':{
        'labelFontSize':15,
        'labelFont':'Courier',
        'labelOverlap':False,
        'symbolStrokeWidth':5,
        'symbolSize':400,
        'titleFontSize':15,
    }
    ,
    'configure_axis':{
        'titleFontSize':20
        , 'labelFontSize':15
    }
    }

def add_properties(chart, title, subtitle, ALTAIR_CSS, faceted_chart=False):
    """
    :param chart: altair chart
    :param title: string major title
    :param subtitle: subtitle
    :param ALTAIR_CSS: use get_altair_css() or similar dict
    :param faceted_chart: boolean to determine if the chart parameter is faceted
    :return: chart with properties added
    """
    if faceted_chart:
        chart = chart.properties(
            title = {'text':[title],'subtitle':[subtitle]}
        ).configure_title(
            fontSize=ALTAIR_CSS['configure_title']['fontSize'],
            font=ALTAIR_CSS['configure_title']['font'],
            anchor=ALTAIR_CSS['configure_title']['anchor'],
            color=ALTAIR_CSS['configure_title']['color']
        ).configure_legend(
            labelFontSize=ALTAIR_CSS['configure_legend']['labelFontSize'],
            labelFont=ALTAIR_CSS['configure_legend']['labelFont'],
            labelOverlap=ALTAIR_CSS['configure_legend']['labelOverlap'],
            symbolSize=ALTAIR_CSS['configure_legend']['symbolSize'],
            symbolStrokeWidth=ALTAIR_CSS['configure_legend']['symbolStrokeWidth'],
            titleFontSize=ALTAIR_CSS['configure_legend']['titleFontSize'],
            title=None
        ).configure_axis(titleFontSize=ALTAIR_CSS['configure_legend']['titleFontSize'])
    else:
        chart = chart.properties(
            width=ALTAIR_CSS['properties']['width'],
            height=ALTAIR_CSS['properties']['height'],
            title = {'text':[title],'subtitle':[subtitle]}
        ).configure_title(
            fontSize=ALTAIR_CSS['configure_title']['fontSize'],
            font=ALTAIR_CSS['configure_title']['font'],
            anchor=ALTAIR_CSS['configure_title']['anchor'],
            color=ALTAIR_CSS['configure_title']['color']
        ).configure_legend(
            labelFontSize=ALTAIR_CSS['configure_legend']['labelFontSize'],
            labelFont=ALTAIR_CSS['configure_legend']['labelFont'],
            labelOverlap=ALTAIR_CSS['configure_legend']['labelOverlap'],
            symbolSize=ALTAIR_CSS['configure_legend']['symbolSize'],
            symbolStrokeWidth=ALTAIR_CSS['configure_legend']['symbolStrokeWidth'],
            titleFontSize=ALTAIR_CSS['configure_legend']['titleFontSize'],
            title=None
        ).configure_axis(titleFontSize=ALTAIR_CSS['configure_legend']['titleFontSize'])
    return chart

def build_fugly_brokeness(data):
    base = alt.Chart(data).mark_line().transform_fold(
        ['c_v', 'u_mass'],
        as_=['Measure', 'Value']
    ).encode(
        color = alt.Color('Measure:N'),
    )
    
    
    cv_chart = alt.Chart(data).mark_line(point=True, color='#1f77b4').encode(
        x='num_topics',
        y='c_v',
        tooltip=['num_topics', 'c_v','alpha','method'],
    )
    umass_chart = alt.Chart(data).mark_line(point=True, color='#ff7f0e').encode(
        x='num_topics',
        y='u_mass',
        tooltip=['num_topics', 'u_mass','alpha','method'],
    )
    line_chart = base + cv_chart + umass_chart
    coherence_chart = add_properties(line_chart, 'Coherence Scores','', get_altair_css()).resolve_scale(y = 'independent')
    return coherence_chart