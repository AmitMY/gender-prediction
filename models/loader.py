#!/bin/python
import os
#sys.path.append("../")

from utils.file_system import savetodir, makedir, listdir

from models.spacy.main import ModelRunner as spacy_runner
from models.pytorch.main import ModelRunner as pytorch_runner
from models.lm_based.main import ModelRunner as kenlm_runner
from models.sklearn_based.main import ModelRunner as sklearn_runner

class ModelLoader:
    ''' A model loader. It takes a list of files and loads them accordingly as models
    '''

    def __init__(self, models_dir):
        ''' Init method for the model runner

            :param models_dir: the directory where all models are stored
        '''
        self.models_dir = models_dir
        
    def load_models(self):
        ''' Method to load all models
        
            :returns: a list of models
        '''
        models = []
        # get the filenames always in alphanumerical order
        filenames = sorted(listdir(self.models_dir, order=name))

        # a bit too brute force but that way we can track what type of models are/can be loaded
        for model_file in filenames:
            ext = os.path.basename(model_file).split('.')[-1]
            model_name = os.path.basename(model_file).split('.')[-2]

            m = None
            if 'lm' in ext:     # language model files
                m = kenlm_runner.ModelRunner(model_name)
                m.load(model_file)
            elif 'sk' in ext:   # scikit learn model files 
                m = sklearn_runner.ModelRunner(model_name)
                m.load(model_file)
            elif 'sp' in ext:   # spacy model files
                m = spacy_runner.ModelRunner(model_name)
                m.load(model_file)
            elif 'tr' in ext:   # pytorch model files
                m = pytorch_runner.ModelRunner(model_name)
                m.load(model_file)

            models.append(m)            
            
        return models
            
            
            
            
            
            
            
            
            
            
            
