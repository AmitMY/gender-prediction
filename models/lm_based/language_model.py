import os
import subprocess
import kenlm


class LanguageModel():
    def __init__(self, text_file=None, ngram=3):
        ''' Initialises the language model by setting the data file

            :param text_file: the file with data to build a language model from
        '''
        kenlm.LanguageModel.order = 10
        self.text_file = text_file
        self.ngram = ngram

    def build(self):
        ''' Builds a statistical LM model (KenLM)
        '''
        current_path = os.path.dirname(os.path.abspath(__file__))
        subprocess.call([os.path.join(current_path, 'train_lm_external.sh'), self.text_file, str(self.ngram)])
        return self.text_file + ".blm"

    def load(self, model_file):
        ''' Loads an already trained model

            :param model_file: the file containing the KenLM model
        '''
        # print('Loading file: ' + model_file)

        self.model = kenlm.Model(model_file)

    def score(self, sent):
        ''' Scores a sentence given with the model

            :param sent: the sentence to score
            :returns: the score
        '''
        return self.model.score(sent)
