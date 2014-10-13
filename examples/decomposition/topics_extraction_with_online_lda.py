"""
==================================================
Topics extraction with Latent Dirichlet Allocation
==================================================

This is an example of showing how to extrat topics from
20 newsgroup dataset with OnlineLDA. Also, It shows how 
top words in each topic changed during training.

The code is modified from
"Topics extraction with Non-Negative Matrix Factorization"
example.

"""

# Authors: Chyi-Kwei Yau


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals.six.moves import xrange
from sklearn.decomposition import OnlineLDA


def online_lda_example():
    """
    Example for LDA online update
    """

    def chunks(l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i + n]

    # In default, we set topic number to 10, and both hyperparameter
    # eta and alpha to 0.1 (`1 / n_topics`)
    n_topics = 10
    alpha = 1. / n_topics
    eta = 1. / n_topics

    # chunk_size is how many records we want to use
    # in each online iteration
    chunk_size = 2000
    n_features = 1000
    n_top_words = 10

    print('Example of LDA with online update')
    print("Loading 20 news groups dataset...")
    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    vectorizer = CountVectorizer(
        max_df=0.8, max_features=n_features, min_df=3, stop_words='english')

    lda = OnlineLDA(n_topics=n_topics, alpha=alpha, eta=eta, kappa=0.7,
                    tau=512., n_jobs=-1, n_docs=1e4, random_state=0, verbose=0)

    for chunk_no, doc_list in enumerate(chunks(dataset.data, chunk_size)):
        if chunk_no == 0:
            doc_mtx = vectorizer.fit_transform(doc_list)
            feature_names = vectorizer.get_feature_names()
        else:
            doc_mtx = vectorizer.transform(doc_list)

        # fit model
        print("\nFitting LDA models with online udpate on chunk %d..." %
              chunk_no)
        lda.partial_fit(doc_mtx)
        print("Topics after training chunk %d:" % chunk_no)
        for topic_idx, topic in enumerate(lda.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
        online_lda_example()
