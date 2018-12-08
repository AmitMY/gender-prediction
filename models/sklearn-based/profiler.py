#!/usr/bin/env python
import readdata as RD
import argparse
import logging
import numpy as np
from cross_genre_profiler import CrossGenrePerofiler

def main(train_file=None,test_file=None):
    logging.info('logging')
    fns_gender = ['unigram', 'bigram', 'char','clusters']#,'diminutives']#,'punctuation']#'length','diminutives', 'nouns', 'adjectives']
    X_test_gender, X_test_txt = RD.readData(args.test_file)
    
    X_train_gender, X_train_txt = RD.readData(args.train_file)
    p_gender = CrossGenrePerofiler(lang='nl', method='logistic_regression', features=fns_gender)
    print(X_train_gender[0])
    p_gender.train(X_train_txt, X_train_gender)
    Y_pred_gender = p_gender.predict(X_test_txt)

    acc = np.mean([1 if Y_pred_gender[i] == X_test_gender[i] else 0 for i in range(len(X_test_gender))])
    print(str(acc))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CLIN shared task gender prediction')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')

    argparser.add_argument('-i', '--train-file', dest='train_file', type=str, required=True,
                           help='Path to the corpus for which the gender and age of the authors have to be predicted')

    argparser.add_argument('-t', '--test-file', dest='test_file', type=str, required=False,
                           help='Path to the corpus for which the gender and age of the authors have to be predicted')

    argparser.add_argument('-o', '--tira_output', dest='tira_output', type=str, required=True,
                           help='Output directory')

    args = argparser.parse_args()
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)
    main(args.train_file, args.test_file)
