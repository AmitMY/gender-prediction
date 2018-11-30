import argparse
import os
import subprocess

from models.lm_based.language_model import LanguageModel
from utils.file_system import rmdir
from collections import OrderedDict
import numpy as np


def split_by_class_data(text, label):
    ''' Splits the text based on the different classes

        :param text: the input text containing text entries
        :param label: the cass labels
        :returns: A dictionary with text and labels, placed under the same label
    '''
    split_data = {}

    for t, l in zip(text, label):
        t = t.rstrip().lstrip()
        if l not in split_data:
            split_data[l] = t.split('\n')
        else:
            split_data[l].extend(t.split('\n'))
    return split_data


def split_by_sent(text, classes):
    ''' Splits the text on newlines.

        :param text: the input text containing text entries
        :param classes: the class labels
        :returns: A dictionary with text and labels
    '''
    sentences = []
    labels = []

    for d, l in zip(text, classes):
        for sent in d.split('\n'):
            sentences.append(sent)
            labels.append(l)
    return sentences, labels


def compare(lm_models, sents):
    ''' For each sentence in dev_set, computes the score according to each language model and compares them.

        :param lm_models: a dictionary of language models
        :param sents: sentences to test with
        :returns: a dictionary of scores with the sentence index as key and value: [predicted class, score]
    '''
    scores = OrderedDict()
    count = 0
    for class_text in lm_models:
        model = LanguageModel()
        model.load(lm_models[class_text])

        for i in range(len(sents)):
            score = model.score(sents[i].rstrip().lstrip())
            if i not in scores:
                scores[i] = [class_text, score]
            else:  # if it has already been filled then - we compare the scores
                if scores[i][1] < score:
                    scores[i] = [class_text, score]
                else:  # otherwise it doesn't change so we pass
                    pass
    return scores


def compare_file(lm_models, dev_set_file):
    ''' For each sentence in dev_set_file, computes the score according to each language model and compares them.
        :param lm_models: a dictionary of language models
        :param dev_set_file: a file with sentences
        :returns: a dictionary of scores with the sentence index as key and value: [predicted class, score]
    '''
    scores = OrderedDict()
    count = 0
    with open(dev_set_file, 'r') as inFile:
        sents = inFile.readlines()
        print(str(len(sents)))
        for class_text in lm_models:
            model = LanguageModel()
            model.load(lm_models[class_text])

            for i in range(len(sents)):
                score = model.score(sents[i].rstrip().lstrip())
                if i not in scores:
                    scores[i] = [class_text, score]
                else:  # if it has already been filled then - we compare the scores
                    if scores[i][1] < score:
                        scores[i] = [class_text, score]
                    else:  # otherwise it doesn't change so we pass
                        pass
    return scores


def cleanup():
    ''' Removes all temporary files and folders
    '''
    rmdir('tmp')


def get_labels_from_file(file):
    ''' Retrieving labels from a file

        :param file: the file containing the labels 
        :returns: a list of labels
    '''
    with open(file) as F:
        lines = F.readlines()

    return [line.rstrip().lstrip() for line in lines]


def preprocess_text(text_file):
    ''' Takes a file and preprocesses the file.
        Currently the preprocessor will tokenize and lower case the data.

        :param text_file: The name of the file to be preprocessed
        :returns: The name of the file that containes the preprocessed output (default is file_name + extension .lc-tok)
    '''
    preprocessed_file_name = text_file + ".lc-tok"
    current_path = os.path.dirname(os.path.abspath(__file__))
    subprocess.call([os.path.join(current_path, 'preprocess.sh'), text_file, preprocessed_file_name])
    return preprocessed_file_name


def compute_accuracy(predicted, expected):
    ''' Computes the accuracy of the prediction

        :param predicted: Predicted values
        :param expected: Expected values
        :returns: accuracy score
    '''
    eq = [1 if predicted[i] == expected[i] else 0 for i in range(len(predicted))]
    return np.mean(eq)
