from feature_pipeline import word_unigrams
from feature_pipeline import word_bigrams
from feature_pipeline import char_ngrams
from feature_pipeline import dim_count 
from feature_pipeline import man_count 
from feature_pipeline import woman_count 
from feature_pipeline import article_count
from feature_pipeline import cluster_info

#from feature_pipeline import diminutives
#from feature_pipeline import noun_count
#from feature_pipeline import adjective_count
from feature_pipeline import sent_length

from feature_pipeline import punctuation_features
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from utils2 import get_classifier
from sklearn.pipeline import Pipeline


class CrossGenrePerofiler():
    def __init__(self, lang=None, method=None, features=None):
        fs = []
        if 'unigram' in features:
            fs.append(word_unigrams())
        if 'bigram' in features:
            fs.append(word_bigrams())
        #if 'diminutive' in features:
        #    fs.append(diminutives())
        #if 'nouns' in features:
        #    fs.append(noun_count())
        #if 'adjectives' in features:
        #    fs.append(adjective_count())
        #if 'length' in features:
        #    fs.append(sent_length())
        if 'char' in features:
            fs.append(char_ngrams())
        if 'punctuation' in features:
            fs.append(punctuation_features())
        if 'diminutives' in features:
            fs.append(dim_count())
        if 'mancount' in features:
            fs.append(man_count())
        if 'womancount' in features:
            fs.append(woman_count())
        if 'artcount' in features:
            fs.append(article_count())
        if 'clusters' in features:
            fs.append(cluster_info())

        #if 'punctuation' in features:
        #    fs.append(punctuation_features())        

        fu = FeatureUnion(fs, n_jobs=1)
        self.pipeline = Pipeline([('features', fu),
                                  ('scale', Normalizer()),
                                  ('classifier', get_classifier(method=method))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
