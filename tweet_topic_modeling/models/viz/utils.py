import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

def generate_topic_examples(lda_model, test_corpus):
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