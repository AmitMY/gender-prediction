#!/bin/bash
import os
import lm_classifier as LMC
import utils.file_system as FS
from data.reader import Data
import language_model as LM

class ModelRunner:
    ''' A main model runner for the LM-based models
    '''
    def __init__(self, model, train, dev, opts={}):
        ''' Init method for the model runner

            :param model: the name of the model
            :param train: the train data (an instance of Data)
            :param dev: the dev data (an instance of Data)
            :param opts: a dictionary with all options
        '''
        self.model = model

        self.train_sents, self.train_labels = train.export()
        self.dev_sents = None
        self.dev_labels = None
        if not dev is None:
            self.dev_sents, self.dev_labels = dev.export()
        self.ngram = 3
        if 'ngram' in opts:
            self.ngram = opts['ngram']

        self.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp')
        if 'tmp_dir' in opts:
            self.tmp_dir = opts['tmp_dir']

        self.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp')
        if 'out_dir' in opts:
            self.out_dir = opts['out_dir']

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def train(self):
        ''' Method to train an LM-based model

            :returns: the accuracy of the model on the dev set
        '''
        # 1. let's split the data into classes (2, 3, 4...)
        _train_sents, _train_labels = LMC.split_by_sent(self.train_sents, self.train_labels)
        split_data = LMC.split_by_class_data(_train_sents, _train_labels)
        for class_label in split_data:
            FS.savetodir(self.tmp_dir, split_data[class_label], str(class_label) + '.dat')

        # 2. let's train the language models (kenlm)
        lm_models = {}
        for class_label in split_data:
            print('Training language model for: ' + str(class_label))
            lm = LM.languageModel(os.path.join(self.tmp_dir, str(class_label) + '.dat'), self.ngram)
            model_file = lm.build() #TODO: optimize this code
            lm_models[class_label] = model_file

        # 3. Now the models are saved, let's experiment (if we are given something to play with)
        accuracy = 0.0
        if not self.dev_sents is None:
            _dev_sents, _dev_labels = LMC.split_by_sent(self.dev_sents, self.dev_labels)
            FS.savetodir(self.tmp_dir, _dev_sents, 'dev.dat')
            dev_file_pc = LMC.preprocess_text(os.path.join(self.tmp_dir, 'dev.dat'))
            results = LMC.compare_file(lm_models, dev_file_pc)
            predicted_labels = [results[i][0] for i in results]
            accuracy = LMC.compute_accuracy(predicted_labels, _dev_labels)

        return accuracy


    def test(self, test):
        ''' Method to test an LM-based model

            :param test: the test data (an instance of Data)
            :returns: a result object
        '''
        
        return res

# To test if your model runs at all
if __name__ == '__main__':
    data = Data("All", "train", tokenize=True)
    train, dev = data.split() # Tokenize=False is just faster, but less accurate
    results = {}
    for ngram in [3, 4, 5, 6]:
        inst = ModelRunner(model="LuMi", train=train, dev=dev, opts={'ngram':ngram})
        results[ngram] = inst.train()
        inst.save("checkpoint") # Make sure this doesn't error
        inst.load("checkpoint") # Make sure this doesn't error
    [print(' '.join([str(i), str(results[i])])) for i in results]
