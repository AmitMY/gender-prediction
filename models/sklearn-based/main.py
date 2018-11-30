from __future__ import division
import sys
import argparse

from data.reader import Data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

class ModelRunner:
    def __init__(self, model, train, dev, opt={}):
        
        self.train_sents, self.train_labels = train.export()
        self.dev_sents, self.dev_labels = dev.export()
        
        #if opts["pos"]==True
                     

        self.vectorizer = CountVectorizer(binary=True, lowercase=False, decode_error='replace')
        self.train_features = self.vectorizer.fit_transform(self.train_sents)
        self.opt = opt

        if model == "svm":
             self.model = LinearSVC()
        
        elif model == "log":
             self.model = LogisticRegression(C=.088)
 
        elif model == "rf":
            #if not opts.trees:
            #    trees = 10
            #else:
            #    trees = opts.trees
            self.model = RandomForestClassifier(n_estimators=10)
            self.train_features = self.train_features.toarray()                #train_features = vectorizer.fit_transform(train_text)
        
        elif model == "nb":
            self.model = BernoulliNB(binarize=None)
        
        elif model == "knn":
            self.model = KNeighborsClassifier(n_neighbors=10)
 
        else:
            raise ValueError("Unknown model " + model)


    def load(self, path):
        pass

    def save(self, path):
        pass 

    def train(self):

        self.model.fit(self.train_features, self.train_labels)    
        _dev_sents = self.vectorizer.transform(self.dev_sents)
        
        return self.model.score(_dev_sents,self.dev_labels)



if __name__ == "__main__":
    train = Data("News, Twitter", "train",["news","twitter"], tokenize=False)
    dev = Data("YouTube", "train",["youtube"], tokenize=False)

    for modelName in ['svm','log','rf','nb','knn']:
        runner = ModelRunner(modelName, train, dev, opt={"pos":True})
        print(runner.train())
        #for d in enumerate(runner.train()):
        #    print(d)
