from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(max_features=25000, ngram_range=(1,2), min_df=5):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True
    )
