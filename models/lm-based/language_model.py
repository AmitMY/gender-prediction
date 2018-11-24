import argparse
import os
import subprocess
from utils.file_system import savetodir, rmdir
import kenlm

class languageModel():
    def __init__(self, text_file=None):
        ''' Initialises the language model by setting the data file

            :param text_file: the file with data to build a language model from
        '''
        self._text_file = text_file
        self.model = None

    def build(self):
        ''' Builds a statistical LM model (KenLM)
        '''
        current_path = os.path.dirname(os.path.abspath(__file__))
        subprocess.call([os.path.join(current_path, 'train_lm.sh'), self._text_file])
        return self._text_file + ".blm"

    def load(self, model_file):
        ''' Loads an already trained model

            :param model_file: the file containing the KenLM model
        '''
        print('Loading file: ' + model_file)

        self.model = kenlm.Model(model_file)

    def score(self, sent):
        ''' Scores a sentence given with the model

            :param sent: the sentence to score
            :returns: the score
        '''
        return self.model.score(sent)
