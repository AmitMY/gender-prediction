import argparse
import os
import subprocess
from language_model import languageModel
from utils.file_system import savetodir, rmdir
from collections import OrderedDict
import numpy as np

def split_by_class(text_file, label_file):
    ''' Splits the text file based on the different classes

        :param text_file: the input file containing text entries
        :param label_file: the input file containing class labels
        :returns: A dictionary with text and labels, placed under the same label
    '''
    split_data = {}

    with open(text_file, 'r') as textIn, open(label_file, 'r') as labelIn:
        for t, l in zip(textIn, labelIn):
            t = t.rstrip().lstrip()
            l = l.rstrip().lstrip()
            if l not in split_data:
                split_data[l] = [t.rstrip().lstrip()]
            else:
                split_data[l].append(t.rstrip().lstrip())
    return split_data

def compare(lm_models, dev_set_file):
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
            model = languageModel()
            model.load(lm_models[class_text])
            
            for i in range(len(sents)):
                score = model.score(sents[i].rstrip().lstrip())
                if i not in scores:
                    scores[i] = [class_text, score]
                else:    # if it has already been filled then - we compare the scores
                    if scores[i][1] < score:
                        scores[i] = [class_text, score]
                    else: # otherwise it doesn't change so we pass
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
    correct = 0.0
    for p, e in zip(predicted, expected):
        if p == e:
            correct += 1.0

    return correct/len(predicted)

def compute_accuracy_np(predicted, expected):
    print("Using NP")
    print(str(len(predicted)))
    print(str(len(expected)))
    eq = [1 if predicted[i] == expected[i] else 0 for i in range(len(predicted))]
    return np.mean(eq)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--text-file', required=True, help='The file containing text.')
    parser.add_argument('-l', '--label-file', required=True, help='The file containing labels.')
    parser.add_argument('-o', '--output-dir', required=False, default='tmp', help='The directory where output is stored. Default is tmp.')
    parser.add_argument('-ds', '--dev-set', required=False, help='The dev set data we are experimenting.')
    parser.add_argument('-dl', '--dev-label', required=False, help='The dev set label we are experimenting.')
    parser.add_argument('-c', '--cleanup', required=False, action='store_true', help='If selected, the temporary folder will be removed.')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', help='Verbose mode.')

    args = parser.parse_args()

    # TODO: make 1 and 2 optional or 3 in another script
    # 1. let's split the data into classes (2, 3, 4...)
    split_data = split_by_class(args.text_file, args.label_file)
    for class_text in split_data:
        savetodir(args.output_dir, split_data[class_text], class_text + '.dat')

    # 2. let's train the language models (kenlm)
    lm_models = {}
    for class_text in split_data:
        print('Training language model for: ' + class_text)
        lm = languageModel(os.path.join(args.output_dir, class_text + '.dat'))
        model_file = lm.build() #TODO: optimize this code
        lm_models[class_text] = model_file

    # 3. Now the models are saved, let's experiment (if we are given something to play with)
    if not args.dev_set == None:
        preprocessed_dev_set = preprocess_text(args.dev_set)
        results = compare(lm_models, preprocessed_dev_set)
        predicted_labels = [results[i][0] for i in results]
        if not args.dev_label == None:
            expected_labels = get_labels_from_file(args.dev_label)
            #print('\n'.join([p + '\t' + e for p,e in zip(predicted_labels, expected_labels)]))
            print(compute_accuracy_np(predicted_labels, expected_labels))

if __name__ == "__main__":
    main()


