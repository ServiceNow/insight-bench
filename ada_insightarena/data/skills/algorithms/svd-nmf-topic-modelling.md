
# SVD&NMF Topic Modelling Pipeline

Required Imports

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from scipy import linalg
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
```


Data Preparation



``` python
def prepare_data(categories=None, remove=('headers', 'footers', 'quotes')):
    """Load and prepare the data"""
    print("Loading data...")
    if categories is None:
        categories = ['alt.atheism', 'talk.religion.misc',
                     'comp.graphics', 'sci.space']

    # Load training and test datasets
    newsgroups_train = fetch_20newsgroups(subset='train',
                                        categories=categories,
                                        remove=remove)
    newsgroups_test = fetch_20newsgroups(subset='test',
                                       categories=categories,
                                       remove=remove)

    print(f"Number of training documents: {len(newsgroups_train.data)}")
    print(f"Number of test documents: {len(newsgroups_test.data)}")

    return newsgroups_train, newsgroups_test
```




Data Manipulation



``` python
def transform_data(data, use_tfidf=False):
    """Transform text data into document-term matrix"""
    global vectorizer, vocab
    print("Transforming data...")

    # Initialize vectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = CountVectorizer(stop_words='english')

    # Transform documents to document-term matrix
    vectors = vectorizer.fit_transform(data.data).todense()
    vocab = np.array(vectorizer.get_feature_names_out())

    print(f"Document-term matrix shape: {vectors.shape}")
    return vectors
```



Models Implementation



``` python
def implement_models(vectors):
    """Implement both NMF and SVD models"""
    global nmf_model, svd_model
    print("Implementing models...")

    # NMF implementation
    nmf_model = NMF(n_components=num_topics, random_state=1)
    W_nmf = nmf_model.fit_transform(vectors)
    H_nmf = nmf_model.components_

    # SVD implementation
    svd_model = TruncatedSVD(n_components=num_topics, random_state=1)
    W_svd = svd_model.fit_transform(vectors)
    H_svd = svd_model.components_

    return W_nmf, H_nmf, W_svd, H_svd
```



Topic Extraction



``` python
def show_topics(H):
    """Extract top words for each topic"""
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in H])
    return [' '.join(t) for t in topic_words]
```


Visualization


``` python
def visualize_results(H_nmf, H_svd):
    """Visualize topic modeling results"""
    print("\nVisualization of Results:")

    # Plot topic distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # NMF topics
    for i, topic_dist in enumerate(H_nmf):
        ax1.plot(topic_dist, label=f'Topic {i+1}')
    ax1.set_title('NMF Topic Distributions')
    ax1.set_xlabel('Terms')
    ax1.set_ylabel('Weight')
    ax1.legend()

    # SVD topics
    for i, topic_dist in enumerate(H_svd):
        ax2.plot(topic_dist, label=f'Topic {i+1}')
    ax2.set_title('SVD Topic Distributions')
    ax2.set_xlabel('Terms')
    ax2.set_ylabel('Weight')
    ax2.legend()

    plt.tight_layout()
    plt.show()
```



Evaluate Results



``` python
def evaluate_results(H_nmf, H_svd):
    """Evaluate and compare NMF and SVD results"""
    print("\nModel Evaluation:")

    print("\nNMF Topics:")
    nmf_topics = show_topics(H_nmf)
    for idx, topic in enumerate(nmf_topics):
        print(f"Topic {idx + 1}: {topic}")

    print("\nSVD Topics:")
    svd_topics = show_topics(H_svd)
    for idx, topic in enumerate(svd_topics):
        print(f"Topic {idx + 1}: {topic}")

    # Calculate reconstruction error for NMF
    print(f"\nNMF Reconstruction Error: {nmf_model.reconstruction_err_}")

    # Calculate explained variance ratio for SVD
    print(f"SVD Explained Variance Ratio: {svd_model.explained_variance_ratio_.sum():.4f}")

    return nmf_topics, svd_topics
```



Save Results



``` python
def save_results(nmf_topics, svd_topics, output_path):
    """Save topic modeling results to file"""
    results = {
        'nmf_topics': nmf_topics,
        'svd_topics': svd_topics,
        'nmf_reconstruction_error': nmf_model.reconstruction_err_,
        'svd_explained_variance': svd_model.explained_variance_ratio_.sum()
    }

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
```



Main Execution



``` python
def main():
    # Configuration
    global num_topics, num_top_words
    num_topics = 5
    num_top_words = 8
    output_path = 'topic_modeling_results.csv'

    # Step 1: Data Preparation
    print("\nPreparing data...")
    train_data, test_data = prepare_data()

    # Step 2: Data Manipulation
    print("\nTransforming data...")
    vectors = transform_data(train_data, use_tfidf=True)

    # Step 3: Implement Models
    print("\nImplementing models...")
    W_nmf, H_nmf, W_svd, H_svd = implement_models(vectors)

    # Step 4: Visualize Results
    print("\nVisualizing results...")
    visualize_results(H_nmf, H_svd)

    # Step 5: Evaluate Results
    print("\nEvaluating results...")
    nmf_topics, svd_topics = evaluate_results(H_nmf, H_svd)

    # Step 6: Save Results
    print("\nSaving results...")
    save_results(nmf_topics, svd_topics, output_path)

if __name__ == "__main__":
    main()
```
