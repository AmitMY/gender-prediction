#!/bin/python
import os
import time
import sys
import shutil
import argparse

from models.lm_based.language_model import LanguageModel
from models.lm_based.lm_classifier import split_by_sent, split_by_class_data, preprocess_text, compare_file, \
    compute_accuracy

sys.path.append("../")

from data.reader import Data

from utils.file_system import savetodir, makedir


class ModelRunner:
    ''' A main model runner for the LM-based models
    '''

    def __init__(self, model, train, dev, opt={}):
        ''' Init method for the model runner

            :param model: the name of the model
            :param train: the train data (an instance of Data)
            :param dev: the dev data (an instance of Data)
            :param opt: a dictionary with all options
        '''
        self.lm_models = {}
        self.model = model

        self.train_sents, self.train_labels, _ = train.export()
        self.dev_sents = None
        self.dev_labels = None
        if not dev is None:
            self.dev_sents, self.dev_labels, _ = dev.export()
        self.ngram = 3
        if 'ngram' in opt:
            self.ngram = opt['ngram']

        microtime = int(round(time.time() * 1000))
        self.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp_' + str(microtime))
        if 'tmp_dir' in opt:
            self.tmp_dir = opt['tmp_dir']

        makedir(self.tmp_dir)

        self.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp_' + str(microtime))
        if 'out_dir' in opt:
            self.out_dir = opt['out_dir']

        makedir(self.out_dir)

    def save(self, filename):
        ''' Saves the whole directory containing the language models by creating a zip archive

            :param filenam: the name of the file to save the archive to
        '''
        shutil.make_archive(filename, 'zip', self.tmp_dir)

    def load(self, filename):
        ''' Unzips a file and loads the language models

            :param filename: the name of the file containing the archive to unpack
    	'''
        shutil.unpack_archive(filename + ".zip", self.tmp_dir)
        class_label = 0
        lm_file = os.path.join(self.tmp_dir, str(class_label) + ".dat.blm")
        while os.path.exists(lm_file):
            self.lm_models[class_label] = lm_file
            class_label += 1
            lm_file = os.path.join(self.tmp_dir, str(class_label) + ".dat.blm")

    def train(self):
        ''' Method to train an LM-based model

            :returns: the accuracy of the model on the dev set
        '''
        # 1. let's split the data into classes (2, 3, 4...)
        _train_sents, _train_labels = split_by_sent(self.train_sents, self.train_labels)
        split_data = split_by_class_data(_train_sents, _train_labels)
        for class_label in split_data:
            savetodir(self.tmp_dir, split_data[class_label], str(class_label) + '.dat')

        # 2. let's train the language models (kenlm)
        for class_label in split_data:
            print('Training language model for: ' + str(class_label))
            lm = LanguageModel(os.path.join(self.tmp_dir, str(class_label) + '.dat'), self.ngram)
            model_file = lm.build()  # TODO: optimize this code
            self.lm_models[class_label] = model_file

        # 3. Now the models are saved, let's experiment (if we are given something to play with)
        accuracy = self.test()
        return accuracy

    def test(self, test=None):
        ''' Method to test an LM-based model

            :param test: the test data (an instance of Data)
            :returns: a result object
        '''
        accuracy = 0.0
        if test is None:
            test_sents = self.dev_sents
            test_labels = self.dev_labels
        else:
            test_sents, test_labels, _ = test.export()

        _test_sents, _test_labels = split_by_sent(test_sents, test_labels)
        savetodir(self.tmp_dir, _test_sents, 'test.dat')
        test_file_pc = preprocess_text(os.path.join(self.tmp_dir, 'test.dat'))
        results = compare_file(self.lm_models, test_file_pc)
        predicted_labels = [results[i][0] for i in results]
        accuracy = compute_accuracy(predicted_labels, _test_labels)

        return [accuracy]

    def cleanup(self):
        ''' Method to cleanup the mess - removes the temp directory
        '''
        rmdir(self.tmp_dir)


# To test if your model runs at all
if __name__ == '__main__':
    ''' read arguments from the command line and train or test a language model based classifier.
    '''

    parser = argparse.ArgumentParser(description='An LM-based classifier.')
    parser.add_argument('-m', '--model', required=False, default=None, help='a model to load.')

    args = parser.parse_args()

    train, dev = Data("Twitter", "train", ['twitter'], tokenize=True).split()
    # dev = Data("N", "train", ['twitter'], tokenize=True)
    results = {}

    if args.model is not None:
        inst = ModelRunner(model="LuMi", train=train, dev=dev, opt={})
        inst.load(args.model)
        results[args.model] = inst.test()
    else:
        for ngram in [3, 4, 5, 6]:
            inst = ModelRunner(model="LuMi", train=train, dev=dev, opt={'ngram': ngram})
            print("Created model", "training...")
            results[ngram] = inst.train()
            inst.save("checkpoint_" + str(ngram))  # Make sure this doesn't error
            # inst.load("checkpoint")  # Make sure this doesn't error

    [print(' '.join([str(i), str(results[i])])) for i in results]
