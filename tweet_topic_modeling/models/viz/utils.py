import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import CoherenceModel, LdaModel
import random
import langid
import altair as alt
import numpy as np
import pandas as pd
from IPython.display import display
import altair as alt
import joblib
from collections import defaultdict

plt.rcParams["figure.figsize"] = [10,5]

def get_replaced_tweets():
    if 'replaced_tweets' not in locals() and 'replaced_tweets' not in globals(): 
        try:
            replaced_tweets = joblib.load("data/whimsy_tweets.joblib")
        except FileNotFoundError:
            raise Exception("Error: File not found.")
        except Exception as e:
            raise Exception(f"{e}")
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
    
def print_results(lda_model, tweet_texts, num_topics, unique_tweet_texts, train_texts, train_tokens, test_texts, dictionary, test_corpus, train_corpus):
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
        print("Test tweet:", unique_tweet_texts[i])
        topic_distribution = lda_model.get_document_topics(doc)
        for topic, prob in topic_distribution:
            print("Topic label:", topic, "\nProbability:", prob)
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

# def generate_word_clouds(lda_model):
#     # Generate word clouds for each topic
#     for idx, topic in lda_model.show_topics(formatted=False):
#         word_freq = {word: freq for word, freq in topic}
#         wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_freq)
#         plt.figure()
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.title('Topic {}'.format(idx))
#         plt.axis('off')
#         plt.show()
def generate_word_clouds(lda_model, clouds_per_row=3):
    num_topics = lda_model.num_topics
    num_rows = (num_topics + clouds_per_row - 1) // clouds_per_row  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, clouds_per_row, figsize=(15, 5*num_rows))

    for idx, topic in lda_model.show_topics(formatted=False):
        word_freq = {word: freq for word, freq in topic}
        wordcloud = WordCloud(background_color='white', contour_color='steelblue').generate_from_frequencies(word_freq)
        row = idx // clouds_per_row  # Calculate the row index
        col = idx % clouds_per_row    # Calculate the column index
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title('Topic {}'.format(idx))
        ax.axis('off')
    
    # Remove empty subplots if there are fewer topics than the maximum number of clouds per row
    if num_topics % clouds_per_row != 0:
        for ax in axes.flat[num_topics:]:
            ax.remove()

    plt.tight_layout()
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

def add_multiline_properties(chart, title, subtitle, ALTAIR_CSS, faceted_chart=False):
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
        ).configure_axis(titleFontSize=ALTAIR_CSS['configure_legend']['titleFontSize']
                         , titleFont='Courier'
                         ,          
        ).configure_axisLeft(labelColor='black', titleColor='black', titleFont='Courier'
        ).configure_axisRight(labelColor='#2ca02c', titleColor='#2ca02c', titleFont='Courier'
        ).configure_axisBottom(labelColor=ALTAIR_CSS['configure_title']['color']
                               , titleColor = ALTAIR_CSS['configure_title']['color']
                               , titleFont='Courier'
        )
        
    return chart
    
def build_multiline_altair(data):
    base = alt.Chart(data).mark_line().transform_fold(
        ['c_v', 'u_mass'],
        as_=['Measure', 'Value']
    ).encode(
        color = alt.Color('model_name:N'),
    )
    
    
    cv_chart = base.mark_line(point=True, opacity=0.90).encode(
        x='num_topics',
        y=alt.Y('c_v', title='c_v score'),
        #color = alt.Color('model_name:N', scale=alt.Scale(scheme='category10')),
        tooltip=['c_v','model_name', 'num_topics'],
    )
    umass_chart = base.mark_line(point=True, opacity=1).encode(
        x='num_topics',
        y=alt.Y('u_mass', title='u_mass score'),
        #color = alt.Color('model_name:N', scale=alt.Scale(scheme='dark2')),
        tooltip=['u_mass','model_name', 'num_topics'],
        #shape=alt.Shape('model_name', scale=alt.Scale(range=['cross']))
    )
    umass_circles = base.mark_circle(
        fillOpacity=0,
        stroke='#2ca02c',
        size=70
    ).encode(
        x=alt.X('num_topics', ),
        y=alt.Y('u_mass', axis=alt.Axis(labels=False), title=""),
        tooltip=['u_mass','model_name', 'num_topics'],
        #size='sum(count):Q',
    )
    cv_circles = base.mark_circle(
        fillOpacity=0,
        stroke='black',
        size=70
    ).encode(
        x=alt.X('num_topics', scale=alt.Scale(domain=[2, 20.9])),
        y=alt.Y('c_v', axis=alt.Axis(labels=False), title=""),
        tooltip=['c_v','model_name', 'num_topics'],
        #size='sum(count):Q',
    )
    line_chart = base + cv_chart + umass_chart + umass_circles + cv_circles
    coherence_chart = add_multiline_properties(line_chart, "Topic Model(s) Coherence Scores","", get_altair_css()).resolve_scale(y = 'independent')
                                                                                                                                      #, color='independent'
                                                                                                                                    #, shape='independent')
    return coherence_chart


def build_multiline_pop(data, selection = alt.selection_multi(fields=['model_name'], bind='legend')):
    
    base = alt.Chart(data).mark_line().transform_fold(
        ['c_v', 'u_mass'],
        as_=['Measure', 'Value']
    ).encode(
        color = alt.Color('model_name:N'),
    )
    
    
    cv_chart = base.mark_line(point=True, opacity=0.90).encode(
        x='num_topics',
        y=alt.Y('c_v', title='c_v score'),
        #color = alt.Color('model_name:N', scale=alt.Scale(scheme='category10')),
        tooltip=['c_v','model_name', 'num_topics'],
        opacity = alt.condition(selection, alt.value(1), alt.value(0.2))
    )
    umass_chart = base.mark_line(point=True, opacity=1).encode(
        x='num_topics',
        y=alt.Y('u_mass', title='u_mass score'),
        #color = alt.Color('model_name:N', scale=alt.Scale(scheme='dark2')),
        tooltip=['u_mass','model_name', 'num_topics'],
        opacity = alt.condition(selection, alt.value(1), alt.value(0.2))
        #shape=alt.Shape('model_name', scale=alt.Scale(range=['cross']))
    )
    umass_circles = base.mark_circle(
        fillOpacity=0,
        stroke='#2ca02c',
        size=70
    ).encode(
        x=alt.X('num_topics', ),
        y=alt.Y('u_mass', axis=alt.Axis(labels=False), title=""),
        tooltip=['u_mass','model_name', 'num_topics'],
        #size='sum(count):Q',
    )
    cv_circles = base.mark_circle(
        fillOpacity=0,
        stroke='black',
        size=70
    ).encode(
        x=alt.X('num_topics', scale=alt.Scale(domain=[2, 20.9])),
        y=alt.Y('c_v', axis=alt.Axis(labels=False), title=""),
        tooltip=['c_v','model_name', 'num_topics'],
        #size='sum(count):Q',
    )
    line_chart = base + cv_chart + umass_chart + umass_circles + cv_circles
    coherence_chart = add_multiline_properties(line_chart, "Topic Model(s) Coherence Scores","", get_altair_css()).resolve_scale(y = 'independent')
                                                                                                                                      #, color='independent'
    coherence_chart = coherence_chart.add_selection(selection)                                                                                                               #, shape='independent')
    return coherence_chart


def get_pyLDAvis_input(method = 'm2', alpha = 'symmetric', time_str = '20240317-132003', num_topics = 9):
    lda_model = LdaModel.load(f'data/lda/lda_{method}{alpha}{str(num_topics)}_{time_str}.model')
    train_corpus = joblib.load(f'data/lda/train_corpus_m2_{time_str}.joblib')
    dictionary = joblib.load(f'data/lda/dictionary_m2_{time_str}.joblib')
    df_data = pd.DataFrame(joblib.load(f'data/df_data_baseline_{time_str}.joblib'))
    logic = ((df_data["method"]==method) & (df_data['num_topics']==num_topics) & (df_data['alpha'] == 'symmetric'))
    coherence_scores_umass = df_data[logic]['coherence_scores_umass'].values[0]
    coherence_scores_cv = df_data[logic]['coherence_scores_cv'].values[0]
    
    return lda_model, train_corpus, dictionary, coherence_scores_umass, coherence_scores_cv

def create_mapping(unique_tweet_texts, tweet_texts, text_preper):
    """
    Create a mapping from unique tweet texts to tweet texts.
    
    Args:
    - unique_tweet_texts (list): List of unique preprocessed tweet texts.
    - tweet_texts (list): List of original tweet texts.
    - text_preper (func): Text preprocessor 
    
    Returns:
    - mapping (dict): Mapping from unique tweet texts to tweet texts.
    """
    mapping = defaultdict(list)
    for tweet_text in tweet_texts:
        preprocessed_text = text_preper(tweet_text)
        if preprocessed_text in unique_tweet_texts:
            mapping[preprocessed_text].append(tweet_text)
    return mapping

def print_original_tweets_for_topics(lda_model, unique_tweet_texts, mapping, dictionary):
    """
    Print one original tweet for each topic along with its topic label, probability, and percentage of unique tweets.
    
    Args:
    - lda_model: Trained LDA model
    - unique_tweet_texts: List of unique preprocessed tweet texts
    - mapping: Mapping from unique tweet texts to tweet texts
    - dictionary: Gensim dictionary object for converting text to bag-of-words representation
    """
    print("-------------------------------------------------------------")
    print("ORIGINAL tweets with their topic label, probability, and percentage of unique tweets:")
    print("-------------------------------------------------------------")
    
    tweets_printed_per_topic = defaultdict(int)
    total_tweets_per_topic = defaultdict(int)

    for doc in unique_tweet_texts:
        topic_distribution = lda_model[dictionary.doc2bow(doc.split())]
        highest_topic, _ = max(topic_distribution[0], key=lambda x: x[1])
        total_tweets_per_topic[highest_topic] += 1
    
    for doc in unique_tweet_texts:
        topic_distribution = lda_model[dictionary.doc2bow(doc.split())]
        highest_topic, highest_proba = max(topic_distribution[0], key=lambda x: x[1])
        
        if tweets_printed_per_topic[highest_topic] < 1:
            tweets_printed_per_topic[highest_topic] += 1
            print("Topic:", highest_topic)
            original_tweet = mapping[doc][0]  
            print(f"ORIGINAL tweet: {original_tweet}")
            print(f"Topic label: {highest_topic} Probability: {highest_proba}")
            print(f"Percentage of unique tweets: {100 * total_tweets_per_topic[highest_topic] / len(unique_tweet_texts):.2f}%")
            print("-------------------------------------------------------------")