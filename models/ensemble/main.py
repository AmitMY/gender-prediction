#!/bin/python
import os
import time
import sys
import argparse
import pickle

from sklearn import svm
#sys.path.append("../")

from data.reader import Data
from models.loader import Loader as ModelLoader
from utils.file_system import savetodir, makedir


class ModelRunner:
    ''' A main model runner for the Ensemble 
    '''

    def __init__(self, model, train=None, dev=None, opt={}):
        ''' Init method for the model runner

            :param model: the name of the model
            :param train: the train data (an instance of Data, or None in case of loading)
            :param dev: the dev data (an instance of Data, or None in case of loading)
            :param opt: a dictionary with all options
        '''
        self.modelname = model
        self.ensemble_model = svm.NuSVC(gamma='scale')

        self.train_sents = None
        self.train_labels = None
        if train is not None:
            self.train_sents, self.train_labels, self.train_ids = train.export(lowercase=opt["lowercase"])
            
        self.dev_sents = None
        self.dev_labels = None
        if not dev is None:
            self.dev_sents, self.dev_labels, self.dev_ids = dev.export(lowercase=opt["lowercase"])

        self.lowercase = False
        if 'lowercase' in opt:
            self.lowercase = opt['lowercase']

        try:
            # load all models that we want to ensemble
            self.pretrained_models = ModelLoader.load_models(opt['pretrained_models'])
        except:
            print('No pretrained models defined! Cannot ensemble!')
            sys.exit()
            
    def save(self, filename):
        ''' Dumps the file as a pickle objects in the directory

            :param filenam: the name of the file to save the model to
        '''
        filename = str(self.modelname) +'.ensemble.sav'
        pickle.dump((self.modelname, self.ensemble_model), open(filename, 'wb'))

    def load(self, filename):
        ''' Loads a model from a pickle file

            :param filename: the name of the file to load
    	'''
        self.momdelname, self.ensemble_model = pickle.load(open(modelFile, 'rb'))
        
    def train(self):
        ''' Method to train an LM-based model

            :returns: the accuracy of the model on the dev set
        '''
        # 1. let's split the data into sentences (2, 3, 4...)
        _train_sents, _train_labels, _train_ids = split_by_sent(self.train_sents, self.train_labels)
        # 2. For each sentence get its scores from the other models
        _train_vectors = [getVector(_train_sent) for _train_sent in _train_sents]

        # 3. Now use this output (from 2.) as a feature to train the ensemble_model (i.e., the SVM)
        self.ensemble_model.fit(_train_vectors, _train_labels)
        
        # 4. Now the model is trained, let's test its accuracy
        accuracy = self.test()
        return accuracy

    def test(self, test=None):
        ''' Method to test the ensemble model

            :param test: the test data (an instance of Data)
            :returns: a result object
        '''
        accuracy = 0.0
        if test is None:
            test_sents = self.dev_sents
            test_labels = self.dev_labels
            test_ids = self.dev_ids
        else:
            test_sents, test_labels, test_ids = test.export(lowercase=self.lowercase)

        _test_sents, _test_labels, _test_ids = split_by_sent(test_sents, test_labels, test_ids)
        
        predicted_labels = [self.ensemble_model.predict(getVector(_test_sent)) for _test_sent in _test_sents]

        accuracy = compute_accuracy(predicted_labels, _test_labels)

        results = dict(zip(_test_ids, predicted_labels)) #put the predicted labels in a dict keyed by the ids
        return [accuracy], results

    def getVector(self, sent):
        ''' Method to evaluate all models and get their prediction for a given sentence
        
            :param sent: the sentence to test with
        '''
        return [model.test(sent) for model in self.all_models]

# To test if your model runs at all
if __name__ == '__main__':
    ''' read arguments from the command line and train or test a language model based classifier.
    '''

    parser = argparse.ArgumentParser(description='An ensemble classifier.')
    parser.add_argument('-m', '--model-dir', required=False, default=None, help='a model to load.')

    args = parser.parse_args()

    train, dev = Data("Twitter", "train", ['twitter'], tokenize=True).split()
    #dev = Data("N", "train", ['twitter'], tokenize=True)
    results = {}

    if args.model is not None:
        inst = ModelRunner(model="Ensemble101", train=train, dev=dev, opt={'lowercase': False})
        inst.load(args.model)
        results[args.model] = inst.test()
    else:
        for ngram in [3, 4, 5, 6]:
            inst = ModelRunner(model="LuMi", train=train, dev=dev, opt={'ngram': ngram, 'lowercase': False})
            print("Created model", "training...")
            results[ngram] = inst.train()
            inst.save("checkpoint_" + str(ngram))  # Make sure this doesn't error
            #inst.load("checkpoint")  # Make sure this doesn't error

    [print(' '.join([str(i), str(results[i])])) for i in results]
