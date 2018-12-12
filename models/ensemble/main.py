#!/bin/python
import os
import time
import sys
import argparse
import pickle

from sklearn import svm
# sys.path.append("../")

from data.reader import Data
from models.loader import Loader as ModelLoader
from utils.file_system import savetodir, makedir


class ModelRunner:
    ''' A main model runner for the Ensemble 
    '''

    def __init__(self, model, models_dir=None, train=None, test=None, opt={}):
        ''' Init method for the model runner

            :param model: the name of the model
            :param train: the train data (an instance of Data, or None in case of loading)
            :param test: the test or dev data (an instance of Data, or None in case of loading)
            :param opt: a dictionary with all options
        '''
        self.modelname = model
        self.ensemble_model = svm.NuSVC(gamma='scale')

        self.train_sents = None
        self.train_labels = None
        if train is not None:
            self.train_sents, self.train_labels, self.train_ids = train.export(
                lowercase=opt["lowercase"])

        self.test_sents = None
        self.test_labels = None
        if not test is None:
            self.test_sents, self.test_labels, self.test_ids = test.export(
                lowercase=opt["lowercase"])

        if models_dir is not None:
            # load all models that we want to ensemble
            self.pretrained_models = ModelLoader.load_models(models_dir)

    def save(self, filename):
        ''' Dumps the file as a pickle objects in the directory

            :param filenam: the name of the file to save the model to
        '''
        filename = str(self.modelname) + '.ensemble.sav'
        pickle.dump((self.modelname, self.ensemble_model),
                    open(filename, 'wb'))

    def load(self, filename):
        ''' Loads a model from a pickle file

            :param filename: the name of the file to load
        '''
        self.momdelname, self.ensemble_model = pickle.load(
            open(modelFile, 'rb'))

    def train(self):
        ''' Method to train an LM-based model

            :returns: the accuracy of the model on the dev set
        '''
        # 1. let's split the data into sentences (2, 3, 4...)
        _train_sents, _train_labels, _train_ids = split_by_sent(
            self.train_sents, self.train_labels)
        # 2. For each sentence get its scores from the other models
        _train_vectors = [getVector(_train_sent)
                          for _train_sent in _train_sents]

        # 3. Now use this output (from 2.) as a feature to train the ensemble_model (i.e., the SVM)
        self.ensemble_model.fit(_train_vectors, _train_labels)

        # 4. Now the model is trained, let's test its accuracy
        accuracy, _results = self.test()
        return accuracy

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

        predicted_labels = [self.ensemble_model.predict(
            getVector(_test_sent)) for _test_sent in _test_sents]

        accuracy = compute_accuracy(predicted_labels, _test_labels)

        # put the predicted labels in a dict keyed by the ids
        results = dict(zip(_test_ids, predicted_labels))
        return [accuracy], results

    def getVector(self, sent):
        ''' Method to evaluate all models and get their prediction for a given sentence

            :param sent: the sentence to test with
        '''
        return [model.test(sent) for model in self.pretrained_models]


# To test if your model runs at all
if __name__ == '__main__':
    ''' read arguments from the command line and train or test an ensemble SVM model.
    '''

    parser = argparse.ArgumentParser(description='An ensemble classifier.')
    parser.add_argument('-m', '--models-dir', default=None,
                        help='a directory with models to load.')
    parser.add_argument('-d', '--train-data', nargs='+', required=False, default=None,
                        help='a list of indicators for the data to use for training.')
    parser.add_argument('-t', '--test-data', required=False,
                        default=None, help='an indicator for the test data.')
    parser.add_argument('-e', '--mode', required=False, default='train',
                        help='mode, indicating whether we train or evaluate ["train"/"test"]')
    args = parser.parse_args()

    train = None
    if args.train_data is not None:
        train = Data(" ".join(args.train_data), "train",
                     args.train_data, tokenize=True)

    test = None
    if args.test_data is not None:
        test = Data("Test", "test", args.test_data, tokenize=True)

    models_dir = None
    if args.models_dir is not None:
        models_dir = args.models_dir

    results = {}

    ens = ModelRunner(model="Ensemble", mode=args.mode,
                      models_dir=models_dir, train=train, dev=test)
    if args.mode == "train":
        print(ens.train())
    elif args.mode == 'test':
        results = ens.test()
