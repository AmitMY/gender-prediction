#!/bin/python
from utils.file_system import savetodir, makedir
from data.reader import Data
from collections import OrderedDict
import os
import time
import sys
import argparse
import pickle
import numpy as np

from sklearn import svm
sys.path.append("../")
#from models.loader import Loader as ModelLoader


class ModelRunner:
    ''' A main model runner for the Ensemble
    '''

    def __init__(self, model, models_dir, test, opt={}):
        ''' Init method for the model runner

            :param model: the name of the model
            :param train: the train data (an instance of Data, or None in case of loading)
            :param test: the test or dev data (an instance of Data, or None in case of loading)
            :param opt: a dictionary with all options
        '''
        self.modelname = model
        self.ensemble_model = svm.NuSVC(gamma='scale')

        self.test_sents, self.test_labels, self.test_ids = test.export(
            lowercase=opt["lowercase"])

        # load all models that we want to ensemble
        self.pretrained_models = ModelLoader.load_models(models_dir)

    def test(self, test=None):
        ''' Method to test the ensemble model

            :param test: the test data (an instance of Data)
            :returns: a result object
        '''

        def compute_accuracy(predicted, expected):
            ''' Computes the accuracy of the prediction

                :param predicted: Predicted values
                :param expected: Expected values
                :returns: accuracy score
            '''

            eq = [1 if predicted[i] == expected[i]
                  else 0 for i in range(len(predicted))]
            return np.mean(eq)

        def predict(sent, weights=None):
            ''' Method to evaluate all models and get their prediction for a given sentence

                :param sent: the sentence to test with
                :param weights: a list of weights to add on the average
                :returns: the prediction 0/1 or M/F
            '''

            vector = [model.test(sent) for model in self.pretrained_models]
            if weights is not None:
                # convert to a -1,1 scale
                vector = np.multiply(np.subtract(vector, 0.5), 2)
                vector = np.multiply(vector, weights)  # add weights
                # convert back to 0, 1 scale
                vector = np.multiply(np.add(vector, 0.5), 0.5)

            prediction = [0.0 if np.average(vector) < 0.5 else 1.0]
            return prediction

        accuracy = 0.0
        if test is None:
            test_sents = self.test_sents
            test_labels = self.test_labels
            test_ids = self.test_ids
        else:
            test_sents, test_labels, test_ids = test.export(
                lowercase=self.lowercase)

        _test_sents, _test_labels, _test_ids = split_by_sent(
            test_sents, test_labels, test_ids)

        predicted_labels = [predict(_test_sent) for _test_sent in _test_sents]

        accuracy = compute_accuracy(predicted_labels, _test_labels)

        # put the predicted labels in a dict keyed by the ids
        results = dict(zip(_test_ids, predicted_labels))
        return [accuracy], results


# To test if your model runs at all
if __name__ == '__main__':
    ''' read arguments from the command line and train or test an ensemble SVM model.
    '''

    parser = argparse.ArgumentParser(description='An ensemble classifier.')
    parser.add_argument('-m', '--models-dir',
                        help='a directory with models to load.')
    parser.add_argument('-t', '--test-data',
                        help='an indicator for the test data.')

    args = parser.parse_args()

    test = Data("Test", "test", args.test_data, tokenize=True)

    results = OrderedDict

    ens = ModelRunner(model="Ensemble", models_dir=models_dir, test=test)
    results = ens.test()
