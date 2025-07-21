

# LDA Implementation Pipeline


Required Imports



``` python
# Required Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pickle
```


Data Preparation



``` python
# Data Preparation
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['text']
```



Data Manipulation



``` python
# Data Manipulation
def preprocess_data(data):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(data)
    return dtm, vectorizer
```



LDA Implementation



``` python
# Model Implementation
def train_lda(data, n_topics=5):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(data)
    return lda
```



Visualization



``` python
# Visualization
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}', fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, w=0.90, hspace=0.3)
    plt.show()
```



Evaluate Results



``` python
# Evaluate Results
def evaluate_model(model, data):
    # Log-Likelihood: Higher the better
    print("Log-Likelihood: ", model.score(data))
    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", model.perplexity(data))
```



Save Results



``` python
# Save Results
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
```



Main Execution



``` python
# Main Execution
def main():
    data = load_data('documents.csv')
    dtm, vectorizer = preprocess_data(data)
    lda_model = train_lda(dtm, n_topics=5)
    plot_top_words(lda_model, vectorizer.get_feature_names_out(), n_top_words=15, title='Topics in LDA Model')
    evaluate_model(lda_model, dtm)
    save_model(lda_model, 'lda_model.pkl')
if __name__ == '__main__':
    main()
```

