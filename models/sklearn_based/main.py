from __future__ import division
import sys
import os
import argparse
import importlib
import numpy as np 
import pickle

#mod_pos = importlib.import_module("models.sklearn_based")
#mod_cross = importlib.import_module("models.sklearn_based.cross_genre_profiler")
#mod_feat = importlib.import_module("models.sklearn_based.feature_pipeline")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

from data.reader import Data
from models.sklearn_based.cross_genre_profiler import CrossGenrePerofiler



class ModelRunner:
    def __init__(self, model, train, dev, opt={}):
        self.train_sents, self.train_labels = train.export()
        self.dev_sents, self.dev_labels = dev.export()
        if opt["fns_gender"] == True:
            fns_gender = ['char','clusters','mancount']#,'diminutives','mancount','womancount']#'unigram', 'bigram', 'char','clusters']#,'clusters'] #'diminutives','mancount','womancount']#,'artcount'] #,'artcount','punctuation','clusters']        

        
        if opt["pos"] == True:
            irint(str(fns_gender))
            #interleaves words withs postags ["I pronoun am verb a det human noun"] 
            self.train_sents=mod.postag(self.train_sents)
            self.dev_sents=mod.postag(self.dev_sents)

        self.vectorizer = CountVectorizer(binary=True, lowercase=False, decode_error='replace')
        self.train_features = self.vectorizer.fit_transform(self.train_sents)
        self.opt = opt
        self.modelname = model        

        if self.modelname == "svm":
             self.model = LinearSVC()
        
        elif self.modelname == "log":
             self.model = LogisticRegression(C=.088,solver='lbfgs')
 
        elif self.modelname == "rf":
            self.model = RandomForestClassifier(n_estimators=10)
            self.train_features = self.train_features.toarray()                #train_features = vectorizer.fit_transform(train_text)
        
        elif self.modelname == "nb":
            self.model = BernoulliNB(binarize=None)
        
        elif self.modelname == "knn":
            self.model = KNeighborsClassifier(n_neighbors=10)

        elif self.modelname == "log-feat":
            self.model = CrossGenrePerofiler(lang='nl', method='logistic_regression', features=fns_gender)
        elif self.modelname == "LoadedModel":
            pass
        else:
            raise ValueError("Unknown model " + self.modelname)


    def load(self, modelFile):
        self.momdelname, self.model = pickle.load(open(modelFile, 'rb'))
         
    def save(self):
        filename = str(self.modelname) +'.sav'
        pickle.dump((self.modelname, self.model), open(filename, 'wb'))
    
    def test(self, test_sents=None, test_labels=None):
        _test_sents = self.dev_sents
        _test_labels = self.dev_labels
        if test_sents is not None:
            _test_sents = test_sents
            _test_labels = test_labels

        if self.modelname == "log-feat":
            predictions = self.model.predict(_test_sents)
            modelscore = str(np.mean([1 if predictions[i] == _test_labels[i] else 0 for i in range(len(_test_labels))]))
        else:
            _test_sents = self.vectorizer.transform(_test_sents)
            modelscore = self.model.score(_test_sents,_test_labels)

        return modelscore


    def train(self):
        
        if self.modelname == "log-feat":
            self.model.train(self.train_sents, self.train_labels)
        else:
            self.model.fit(self.train_features, self.train_labels)    
       
        return self.test()


if __name__ == "__main__":

    ''' read arguments from the command line and train or test a language model based classifier.
    '''

    parser = argparse.ArgumentParser(description='set of sklearn classifier.')
    parser.add_argument('-m', '--model', required=False, default=None, help='a model to load.')

    args = parser.parse_args()
    
    #YOUTUBE#
    train = Data("Twitter, News", "train",["twitter","news"], tokenize=False)
    dev = Data("YouTube", "train",["youtube"], tokenize=False)

    #TWITTER#
    #train = Data("YouTube, News", "train",["youtube","news"], tokenize=False)
    #dev = Data("Twitter", "train",["twitter"], tokenize=False)

    #NEWS#
    #train = Data("YouTube, Twitter", "train",["youtube","twitter"], tokenize=False)
    #dev = Data("News", "train",["news"], tokenize=False)
   
    if args.model is not None:
        runner = ModelRunner('LoadedModel', train, dev, opt={"pos":False , "fns_gender":True})
        runner.load(args.model)
        a = runner.test()
        print(a)
    else: 
        for modelName in ['log-feat','svm']:#,'log','nb','knn', 'rf']:
            runner = ModelRunner(modelName, train, dev, opt={"pos":False , "fns_gender":True})
            print(runner.train())
            runner.save()
            #for d in enumerate(runner.train()):
            #    print(d)
