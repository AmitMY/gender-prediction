import sys
import argparse
import numpy as np

from sklearn import svm
sys.path.append("../")
from data.reader import Data
#from models.loader import Loader as ModelLoader

class PredictionRunner:
    ''' A main runner for the Ensemble based on already predicted labels
    '''

    def __init__(self, model, scores_per_model):
        ''' Init method for the model runner

            :param model: the name of the model
            :param test_data: the test or dev data (a dictionary of dictionaries)
            :param opt: a dictionary with all options
        '''
        self.modelname = model

        self.scores_per_id = self.__scores(scores_per_model)

    def __scores(self, scores):
        ''' Reads the test files (per model) and extracts scores per id
            
            :params scores: scores organized per model and id
            :returns: scores organized only per id
        '''
        
        scores_per_id = {}
        for model in scores:
            for id in scores[model]:  
                if id not in scores_per_id:
                    scores_per_id[id] = [float(scores[model][id])]
                else:
                    scores_per_id[id].append(float(scores[model][id]))
        return scores_per_id
                
    def evaluate(self, weights=None, expected=None, k=None):
        ''' Method to test the ensemble model

            :param weights: weights
            :param expected: the expected output
            :returns: accuracy, result with digits, result with labels
        '''
        
        def male_female(score):
            ''' Return the label based on the score
                0 = M
                1 = F

                :param score: the score -a number int or float
                :returns: label
            '''
            return 'M' if score < 0.5 else 'F'
    
    
        def compute_accuracy(predicted, expected):
            ''' Computes the accuracy of the prediction

                :param predicted: Predicted values
                :param expected: Expected values
                :returns: accuracy score
            '''
            predicted_list = list(predicted.values())
            eq = [1 if predicted_list[i] == expected[i]
                  else 0 for i in range(len(predicted.values()))]

            return np.mean(eq)
            
        def predict(vector, weights=None, top_k_weight_idx=None):
            ''' Method to evaluate all models and get their prediction for a given sentence

                :param vector: the vector represnting the sentence to test with
                :param weights: a list of weights to add on the average
                :param top_k_weight_idx: a list of indexes of the top k values from the weights
                :returns: the prediction M/F
            '''
            if top_k_weight_idx is not None:
                vector = [vector[i] for i in top_k_weight_idx]
                
            vector = np.multiply(np.subtract(vector, 0.5), 2)
            if weights is not None:
                if top_k_weight_idx is not None:
                    weights = [weights[i] for i in top_k_weight_idx]
                
                vector = np.multiply(vector, weights)  # add weights
                
            prediction = 0.0 if np.average(vector) < 0.0 else 1.0
            return prediction

        # Actual testing/evaluation
        top_k_weight_idx = None
        accuracy = 0.0
        predicted_labels = {}
        if k is not None and k < len(weights):
            top_k_weight_idx = np.argsort(weights)[-k:]
        for id in self.scores_per_id:
            predicted_labels[id] = predict(self.scores_per_id[id], weights, top_k_weight_idx)
            
        if expected is not None:
            accuracy = compute_accuracy(predicted_labels, expected)

        return accuracy, predicted_labels
        


class ModelRunner:
    ''' A main model runner for the Ensemble
    '''

    def __init__(self, model, model_list, test_data):
        ''' Init method for the model runner

            :param model: the name of the model
            :param model_list: the list of models to ensemble
            :param test: the test or dev data (an instance of Data, or None in case of loading)
            :param opt: a dictionary with all options
        '''
        self.modelname = model
        self.ensemble_model = svm.NuSVC(gamma='scale')

        self.test_sents, self.test_labels, self.test_ids = test_data.export(
            lowercase=False)
        # load all models that we want to ensemble
        self.pretrained_models = model_list

    def evaluate(self, test_data=None, weights=None):
        ''' Method to test the ensemble model

            :param test: the test data (an instance of Data)
            :param weights: weights
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
            vector = [model.eval_one(sent) for model in self.pretrained_models]
            threshold = 0.5
            if weights is not None:
                # convert to a -1,1 scale
                vector = np.multiply(np.subtract(vector, 0.5), 2)
                vector = np.multiply(vector, weights)  # add weights
                threshold = 0.0
                
            prediction = 0.0 if np.average(vector) < threshold else 1.0
            return prediction

        # Actual testing/evaluation
        accuracy = 0.0
        test_sents = self.test_sents
        test_labels = self.test_labels
        test_ids = self.test_ids
        if test_data is not None:
            test_sents, test_labels, test_ids = test_data.export(
                lowercase=False)

        predicted_labels = [predict(test_sent, weights) for test_sent in test_sents]

        accuracy = compute_accuracy(predicted_labels, test_labels)

        # put the predicted labels in a dict keyed by the ids
        results = dict(zip(test_ids, predicted_labels))
        return [accuracy], results


# To test if your model runs at all
if __name__ == '__main__':
    ''' read arguments from the command line and train or test an ensemble SVM model.
    '''

    parser = argparse.ArgumentParser(description='An ensemble classifier.')
    parser.add_argument('-m', '--models', nargs='+',
                        help='a directory with models to load.')
    parser.add_argument('-t', '--test-data',
                        help='an indicator for the test data.')

    args = parser.parse_args()

    test = Data("Test", "test", args.test_data, tokenize=True)

    ens = ModelRunner(model="Ensemble", model_list=args.models, test_data=test)
    _, results = ens.evaluate()
