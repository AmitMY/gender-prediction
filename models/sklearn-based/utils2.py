from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from nltk.corpus import stopwords
import xgboost as xgb


def get_classifier(method='logistic_regression'):
    if 'logistic_regression' == method:
        return LogisticRegression(C=1e3,
                                  tol=0.01,
                                  multi_class='ovr',
                                  solver='liblinear',
                                  n_jobs=-1,
                                  random_state=123)

    if 'svc' == method:
        return LinearSVC(C=1e3,
                         tol=0.01,
                         multi_class='ovr',
                         random_state=123)



    if 'random_forest' == method:
        return RandomForestClassifier(n_estimators=250,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=123)

    if 'gradient_boosting' == method:
        return xgb.XGBClassifier(max_depth=10,
                                 subsample=0.7,
                                 n_estimators=500,
                                 min_child_weight=0.05,
                                 colsample_bytree=0.3,
                                 learning_rate=0.1)


#def get_stopwords(languages=['english', 'dutch', 'spanish']):
#    result = []
#    [result.extend(stopwords.words(lang)) for lang in languages]
#    return result
