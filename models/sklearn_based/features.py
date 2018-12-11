from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
import numpy as np
from string import printable
from models.sklearn_based.tweet_tokenizer import TweetTokenizer

tokenizer = TweetTokenizer()


def tokenize(x):
    x = list(filter(lambda x: x in printable, x))
    return tokenizer(x)


class PunctuationFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(['avg_comma_count', 'avg_dot_count',
                         'avg_exclamationmark_count',
                         'avg_questionmark_count'])

    def fit(self, documents, y=None):
        return self

    def avg_punction_count(self, tokens, character):
        if len(tokens) == 0:
            return 0.0
        trueSum = 0
        for token in tokens:
            if token == character:
                trueSum += 1
        return 1.0 * trueSum / len(tokens)

    def transform(self, documents):
        
        tokens_list = [tokenize(doc) for doc in documents]

        avg_comma = [self.avg_punction_count(tokens, ',')
                     for tokens in tokens_list]

