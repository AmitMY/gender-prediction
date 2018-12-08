#!/bin/python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

import numpy as np
import json
#from spelling_error_profiler import SpellingError
from features import PunctuationFeatures
from text_cleaner import TextCleaner
from cluster_extractor import ClusterExtractor
###### TO ADD NEW FEATURES THAT HAVE NO BUILD-IN, BUILD YOUR OWN TRANSFORMER ################

import json


class ClusterInfo(BaseEstimator):
    def __init__(self,fileName='/home/evanmas2/NMTdata/data/ClinSharedTask/cluster-semantic-vectors/glove_clusters_0.011442950_words.json'):
        self.fileName=fileName
        self.data=self.getClusters()   
  
    def getClusters(self):
        print("get data")
        with open(self.fileName) as json_file:
            data = json.load(json_file)
        print("got data")
        data_inv = {}
        for k, v in data.items():
            for i in v:
                data_inv[i] = k
        return data_inv
    
    def transform(self,documents,y=None):
        clusterL=[]
        print("Look for clusters")
        for doc in documents:
            clusterS = [self.data[s] if s in self.data else "0" for s in doc.split()]               
            #for s in doc.split():
                #cluster=[k for k, v in self.data.items() if s.lower() in v]
                #cluster = [self.data[s] for s in doc.split() if s in self.data else 0]               
                #if s in self.data: 
                #    clusterS.append(self.data[s])
                #else:
                #    clusterS.append("NA")
            
            clusterL.append(" ".join(clusterS)) 
        print("Example cluster")
        print(clusterL[1])
        
        #onehotenc = preprocessing.Multi(handle_unknown='ignore')
        #print("prep onehotenc")
        #print(str(np.array(clusterL).shape()))
        #exit()

        #onehotenc.fit(np.array(clusterL))
        #print("fitting done") 
        #X=onehotenc.transform(clusterL).toarray().T
        #X = np.array(clusterL).T
        #print("X is done") 
        #if not hasattr(self,'scalar'):
        #    self.scalar=preprocessing.StandardScaler().fit(X)
        
        return clusterL
   
    def fit(self, documents, y=None):
        return self
## NOT FINISHED ##

def punctuation_features():
    pipeline = Pipeline([('feature', PunctuationFeatures()),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('punctuation_features', pipeline)

class SentLength(BaseEstimator):
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):        
        length_list = [len(doc.split()) for doc in documents]

        X = np.array([length_list]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)

        return self.scalar.transform(X)

class DiminutiveCount(BaseEstimator):
    
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        dim_endings=["tje","tjes","kje","kjes","pje","pjes","dje","djes","fje","fjes","bje","bjes","gje","gjes","sje","sjes"]
        dim_avg=[]

        for doc in documents:
            dim_count=0
            word_count=0
            for s in doc.split():
                word_count+=1
                if s.endswith(tuple(dim_endings)) and s not in dim_endings:
                    dim_count+=1
            dim_avg.append(dim_count/word_count)
            
        X=np.array([dim_avg]).T
         
        if not hasattr(self,'scalar'):
            self.scalar=preprocessing.StandardScaler().fit(X)
        
        return self.scalar.transform(X)

class ManCount(BaseEstimator):
    
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        man_words=["je","d'r","ja","nee","neen","tamelijk","onmiddellijk","ongelofelijk","feitelijk","duidelijk"]
        dim_avg=[]
        for doc in documents:
            man_count=0
            word_count=0
            for s in doc.split():
                word_count+=1
                if s.lower() in man_words:
                    man_count+=1
            dim_avg.append(man_count/word_count)
            
        X=np.array([dim_avg]).T
         
        if not hasattr(self,'scalar'):
            self.scalar=preprocessing.StandardScaler().fit(X)
        
        return self.scalar.transform(X)

class WomanCount(BaseEstimator):
    
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        woman_words=["ik","hij","dadelijk","vriendelijk","lelijk","vrolijk","eindelijk","verschrikkelijk"]
        dim_avg=[]
        for doc in documents:
            woman_count=0
            word_count=0
            for s in doc.split():
                word_count+=1
                if s.lower() in woman_words:
                    woman_count+=1
            dim_avg.append(woman_count/word_count)
            
        X=np.array([dim_avg]).T
         
        if not hasattr(self,'scalar'):
            self.scalar=preprocessing.StandardScaler().fit(X)
        
        return self.scalar.transform(X)

class ArtCount(BaseEstimator):
    
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        articles=["de","het","een"]
        dim_avg=[]
        for doc in documents:
            article_count=0
            word_count=0
            for s in doc.split():
                word_count+=1
                if s.lower() in articles:
                    article_count+=1
            dim_avg.append(article_count/word_count)
            
        X=np.array([dim_avg]).T
         
        if not hasattr(self,'scalar'):
            self.scalar=preprocessing.StandardScaler().fit(X)
        
        return self.scalar.transform(X)
        
        

def word_unigrams():
    preprocessor = TextCleaner(lowercase=True,
                               filter_urls=True,
                               filter_mentions=True,
                               filter_hashtags=True,
                               alphabetic=True,
                               strip_accents=True,
                               filter_rt=True)
    vectorizer = CountVectorizer(min_df=2,
                                 preprocessor=preprocessor,
                                 ngram_range=(1, 1))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_unigrams', pipeline)


def word_bigrams():
    preprocessor = TextCleaner(lowercase=True,
                               filter_urls=True,
                               filter_mentions=True,
                               filter_hashtags=True,
                               alphabetic=False,
                               strip_accents=False,
                               filter_rt=True)
    pipeline = Pipeline([('vect', CountVectorizer(preprocessor=preprocessor,
                                                  ngram_range=(2, 2))),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_bigrams', pipeline)


def char_ngrams():
    vectorizer = CountVectorizer(min_df=1,
                                 preprocessor=TextCleaner(filter_urls=True,
                                                          filter_mentions=True,
                                                          filter_hashtags=True,
                                                          lowercase=False),
                                 analyzer='char_wb',
                                 ngram_range=(4, 4))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('char_ngrams', pipeline)
                                      

def sent_length():
 
    pipeline = Pipeline([('feature', SentLength()),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    
    return ('sent_length', pipeline)

def man_count():
    pipeline = Pipeline([('feature', ManCount()), 
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('man_count', pipeline)

def woman_count():
    pipeline = Pipeline([('feature', WomanCount()), 
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('woman_count', pipeline)

def article_count():
    pipeline = Pipeline([('feature', ArtCount()), 
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('article_count', pipeline)

def dim_count():
    pipeline = Pipeline([('feature', DiminutiveCount()), 
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('dim_count', pipeline)

def cluster_info():

    preprocessor = ClusterExtractor(lowercase=True,
                               filter_urls=True,
                               filter_mentions=True,
                               filter_hashtags=True,
                               alphabetic=False,
                               strip_accents=False,
                               filter_rt=True)
    
    vectorizer = CountVectorizer(min_df=2,
                                 preprocessor=preprocessor,
                                 ngram_range=(1, 1))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    
    return ('cluster_info', pipeline)


    #getoutput = clusterInfo (X_train)
    #Problem here because the feature list I'm feeding it, doesn't always have the same size  
    #pipeline = Pipeline([('feature', ClusterInfo()), 
    #                     ('tfidf', TfidfTransformer(sublinear_tf=False)),
    #                     ('scale', Normalizer())])
    #return ('cluster_info', pipeline)



