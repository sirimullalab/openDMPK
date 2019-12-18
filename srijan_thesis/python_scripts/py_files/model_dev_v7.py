#######################################################################################
# Author: Srijan Verma                                                              #
# School of Pharmacy                                                                #
# Sirimulla Research Group [http://www.sirimullaresearchgroup.com/]                 #
# The University of Texas at El Paso, TX, USA                                       #
# Last modified: 19/12/2019                                                         #
# Copyright (c) 2019 Srijan Verma and Sirimulla Research Group, under MIT license   #
#######################################################################################
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import time
#Documentation of hypopt - https://www.pydoc.io/pypi/hypopt-1.0.3/autoapi/model_selection/index.html
#Edited version -> added cohen score as metric!
from hypopt import GridSearch
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import ExtraTreesClassifier #Compare with decision tree
from sklearn.gaussian_process import GaussianProcessClassifier
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier #Compare with decision tree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

model_runtime = []
dataset_size = []
train_size = []
val_size = []
test_size = []
train_pos = []
val_pos = []
test_pos = []
models = []
feature_used = []
target = []
accuracy_val = []
TP_val = []
TN_val = []
FP_val = []
FN_val = []
accuracy_test = []
TP_test = []
TN_test = []
FP_test = []
FN_test = []
c_kappa_val = []
LOG_loss_val = []
precision_0_val = []
recall_0_val = []
f1_score_0_val = []
precision_1_val = []
recall_1_val = []
f1_score_1_val = []
auc_out_val = []
auc_prec_rec_val = []
c_kappa_test = []
LOG_loss_test = []
precision_0_test = []
recall_0_test = []
f1_score_0_test = []
precision_1_test = []
recall_1_test = []
f1_score_1_test = []
auc_test_0 = []
auc_test_1 = []
auc_val_0 = []
auc_val_1 = []
arr = []
X_train = []
X_val = []
X_test = []
y_train = []
y_val = []
y_test = []
save_params = []

seed = 7
verbose = 0

##############################<SPLIT DATA INTO TRAIN, VAL AND TEST>##############################
def get_splitted_data(numpy_path_tr, numpy_path_va, numpy_path_te, _test_size, _val_size):
    
    global X_train, X_val, X_test, y_train, y_val, y_test, arr
    
    feature_and_target = (os.path.split(numpy_path_tr)[1][0:len(os.path.split(numpy_path_tr)[1])-7])
    feature_used.append(feature_and_target.split('-')[0])
    target.append(feature_and_target.split('-')[1])
    
    arr = np.load(numpy_path_tr)
    X_train = arr[:,0:(arr.shape[1]-1)]
    y_train = arr[:,(arr.shape[1]-1)]
    
    arr = np.load(numpy_path_va)
    X_val = arr[:,0:(arr.shape[1]-1)]
    y_val = arr[:,(arr.shape[1]-1)]
    
    arr = np.load(numpy_path_te)
    X_test = arr[:,0:(arr.shape[1]-1)]
    y_test = arr[:,(arr.shape[1]-1)]


#    arr = np.load(numpy_path)
#    X = arr[:,0:(arr.shape[1]-3)]
#    Y = arr[:,(arr.shape[1]-3)]
#    FY = arr[:,(arr.shape[1]-2):]
    _val_size = float(_val_size)
    _test_size = float(_test_size)

    if _test_size == 0.0:
        X_test = 0
        y_test = 0
        test_pos.append(0)
        test_size.append(0)
#        X_train, X_val, y_train, y_val = model_selection.train_test_split(X, Y, test_size=(round((_val_size/(1-_test_size)), 3)), random_state=seed, stratify=Y)

    else:
#        success = False
#        while(success==False):
#            try:
#                X_train, X_test, y_train, y_test = model_selection.train_test_split(X, FY, test_size=_test_size, random_state=seed, stratify=FY)
#                X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=(round((_val_size/(1-_test_size)), 3)), random_state=seed, stratify=y_train)
#                y_train = y_train[:,1]
#                y_val = y_val[:,1]
#                y_test = y_test[:,1]

        test_pos.append(np.count_nonzero(y_test))
        test_size.append(X_test.shape[0])
#                success=True

#            except:
#                d = pd.DataFrame(FY, columns=['F', 'Y'])
#
#                d_class = d.groupby(['F','Y']).Y.agg('count').to_frame('class_count').reset_index()
#                d_class_1 = d_class[d_class.Y == 1].reset_index(drop=True,inplace=False)
#                d_class_1.sort_values('class_count',ascending=True,inplace=True)
#                first_low_1 = d_class_1.iloc[0][0]
#                second_low_1 = d_class_1.iloc[1][0]
#
#                d.F.loc[(d['F'] == first_low_1) & (d['Y'] == 1)] = second_low_1
#
##                d_class_0 = d_class[d_class.Y == 0].reset_index(drop=True,inplace=False)
##                d_class_0.sort_values('class_count',ascending=True,inplace=True)
##                first_low_0 = d_class_0.iloc[0][0]
##                second_low_0 = d_class_0.iloc[1][0]
##
##                d.F.loc[(d['F'] == first_low_0) & (d['Y'] == 0)] = second_low_0
#                FY = d.values
#                pass

    train_pos.append(np.count_nonzero(y_train))
    val_pos.append(np.count_nonzero(y_val))

    dataset_size.append(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])

#    dataset_size.append(arr.shape[0])
    train_size.append(X_train.shape[0])
    val_size.append(X_val.shape[0])


    return ( X_train, X_val, X_test, y_train, y_val, y_test )

##############################################################################################################################

##############################<GET SPECIFIC MODEL FUNCTION>##########################################
def get_model_function(func):
    
    function_mappings = {
        'RandomForestClassifier': RandomForestClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'MLPClassifier': MLPClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'MultinomialNB': MultinomialNB(),
        'XGBClassifier': XGBClassifier(),
        'DummyClassifier': DummyClassifier(),
        'GaussianNB': GaussianNB(),
        'SVC': SVC(probability = True),
        'NuSVC': NuSVC(probability = True),
        'BaggingClassifier': BaggingClassifier(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'GaussianProcessClassifier': GaussianProcessClassifier(),
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(),
        'ExtraTreeClassifier': ExtraTreeClassifier(),
        'LinearSVC': LinearSVC(),
        'NearestCentroid': NearestCentroid(),
        'OneVsOneClassifier': OneVsOneClassifier(MultinomialNB()),
        'OneVsRestClassifier': OneVsRestClassifier(LogisticRegression()),
        'OutputCodeClassifier': OutputCodeClassifier(MultinomialNB()),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
        'Perceptron': Perceptron(),
        'RidgeClassifier': RidgeClassifier(),
        'SGDClassifier': SGDClassifier(),
        'BayesianGaussianMixture': BayesianGaussianMixture(),
        'GaussianMixture': GaussianMixture()

    }

    if func == 'AllModels':
        return [*function_mappings]

    else:
        return function_mappings[func]

##############################################################################################################################

##############################<GET MODEL ALL PARAMETERS FOR GRID SEARCH>##########################################
def get_model_params(func):

    #    for classification tasks (https://stackoverflow.com/questions/23939750/understanding-max-features-parameter-in-randomforestregressor)
    random_forest_params = {
                            'n_estimators': [i for i in range(50, 1000, 150)],
#                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None] + [i for i in range(5, 30, 6)],
#                            'min_samples_split': np.linspace(0.2, 2, 10, endpoint=True),
#                            'min_samples_leaf': np.linspace(1, 10, 10, endpoint=True),
                            'min_weight_fraction_leaf': [0.0],
                            'max_features': ['auto', 'log2'],
                            'max_leaf_nodes': [None],
                            'min_impurity_decrease': [0.0],
                            # 'min_impurity_split' :
                            'bootstrap':[False],
                            'oob_score': [False],   #  When data too less, then set this to true : reference -
                            'n_jobs': [-1],   #  https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710
                            'random_state': [seed],
                            'verbose':[verbose],
                            'warm_start': [False],  # For retraining a trained model. Refer -
                            'class_weight': ['balanced', None] #https://stackoverflow.com/questions/42757892/how-to-use-warm-start/54304493
                     # For class weight! V.imp!! - https://chrisalbon.com/machine_learning/trees_and_forests/handle_imbalanced_classes_in_random_forests/
                     		}

    decision_tree_params = {
                                'criterion': ['gini', 'entropy'],
                                'splitter': ['best', 'random'],
                                'max_depth': [None] + [i for i in range(1, 30, 4)],
                                'min_samples_split': np.linspace(0.2, 2.0, 10, endpoint=True),
                                'min_samples_leaf': np.linspace(0.5, 1.5, 5, endpoint=True),
                                'min_weight_fraction_leaf': [0.0],
                                'max_features': ['auto'],
                                'random_state': [seed],
                                'max_leaf_nodes': [None],
                                'min_impurity_decrease': [0.0],
                                'class_weight': ['balanced'],
                                'presort': [False]
                                }

    adaboost_params = {
                        'base_estimator':[None],
                        'n_estimators':[i for i in range(50, 1000, 50)],
                        'learning_rate':[0.01, 0.1, 1],
                        'algorithm':['SAMME.R'],
                        'random_state':[seed]
                        }

    def get_hidden_layers():
        import itertools
        x = [100, 256, 512]
        hl = []

        for i in range(1, len(x)):
            hl.extend([p for p in itertools.product(x, repeat=i+1)])

        return hl

    hidden_layer_sizes = get_hidden_layers()


    mlp_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': ['relu'],
                    'solver': ['adam'],
#                    'alpha': 10.0 ** -np.arange(4, 6),
                    'batch_size': ['auto'],
#                    'learning_rate': ['adaptive'],
#                    'learning_rate_init':[0.01],
                    'power_t':[0.5],
                    'max_iter':[300],
#                    'shuffle': [True],
                    'random_state':[seed],
#                    'tol':[0.0001],
#                    'verbose':[verbose],
#                    'warm_start':[False],
#                    'momentum':[0.9],
#                    'nesterovs_momentum':[True],
#                    'early_stopping':[True],
#                    'validation_fraction':[0.1],
#                    'beta_1':[0.9],
#                    'beta_2':[0.999],
#                    'epsilon':[0.00000001],
                    'n_iter_no_change':[10]
                }# batch_size = min(200, n_samples)

    svc_params  =	{       'penalty': ['l1', 'l2'],
							'C': [0.1, 1, 2, 5, 7, 10],
							'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
							'degree': [0, 1, 2, 3, 4, 5],
							'gamma':[0.0001 ,0.001, 0.01, 0.1, 1],
							'coef0': [0.0], 
							'shrinking': [True],
							'probability':[True], 
							'tol': [1e-4],
							'verbose': [verbose],
							'max_iter': [500],
							'decision_function_shape': ['ovr'], # one-vs-one (‘ovo’) is always used as multi-class strategy 
							'random_state':[seed]				#Refer - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
					}
#BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0

    bagging_classifier_params = {
        
        
                                    'base_estimator': [MLPClassifier(), None],
                                    'n_estimators': [10],
                                    'max_samples': [1.0],
                                    'max_features': [1.0],
                                    'bootstrap': [True],
                                    'oob_score': [False],
                                    'warm_start': [False],
                                    'n_jobs': [-1],
                                    'random_state': [seed],
                                    'verbose': [verbose]
                                
                                }
#LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    logistic_regression_params = {
                                    'penalty': ['l2', 'l1'],
                                    'dual': [False],
                                    'tol': [0.0001],
                                    'C': [1.0],
                                    'fit_intercept': [True],
                                    'intercept_scaling': [1],
                                    'class_weight': ['balanced', None],
                                    'random_state': [seed],
#                                    'solver': ['lbfgs', 'liblinear'],
#                                    'max_iter': [200],
                                    'multi_class': ['ovr'],
                                    'verbose': [verbose],
                                    'warm_start': [False],
                                    'n_jobs': [-1],
                                    'l1_ratio': [None]
                                }
##{'C': 1.0,
#'average': False,
#'class_weight': None,
#'early_stopping': False,
#'fit_intercept': True,
#'loss': 'hinge',
#'max_iter': 1000,
#'n_iter_no_change': 5,
#'n_jobs': None,
#'random_state': None,
#'shuffle': True,
#'tol': 0.001,
#'validation_fraction': 0.1,
#'verbose': 0,
#'warm_start': False}

    passive_aggressive_classifier_params = {
                                            'C': [1.0],
                                            'average': [False],
                                            'class_weight': [None, 'balanced'],
                                            'early_stopping': [False],
                                            'fit_intercept': [True],
                                            'loss': ['hinge'],
                                            'max_iter': [1000],
                                            'n_iter_no_change': [5],
                                            'n_jobs': [-1],
                                            'random_state': [seed],
                                            'shuffle': [True],
                                            'tol': [0.001],
                                            'validation_fraction': [0.1],
                                            'verbose': [verbose],
                                            'warm_start': [False]
                                        }
                                            
#ExtraTreesClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)

    extra_trees_classifier_params =         {
#                                        'n_estimators': [i for i in range(50, 1000, 150)],
                                        #                            'criterion': ['gini', 'entropy'],
#                                        'max_depth': [None] + [i for i in range(5, 30, 6)],
                                        #                            'min_samples_split': np.linspace(0.2, 2, 10, endpoint=True),
#                                        'min_samples_leaf': np.linspace(1, 10, 10, endpoint=True),
                                        'min_weight_fraction_leaf': [0.0],
                                        'max_features': ['auto', 'log2'],
                                        'max_leaf_nodes': [None],
                                        'min_impurity_decrease': [0.0],
                                        # 'min_impurity_split' :
                                        'bootstrap':[False],
                                        'oob_score': [False],   #  When data too less, then set this to true : reference -
                                        'n_jobs': [-1],   #  https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710
                                        'random_state': [seed],
                                        'verbose':[verbose],
                                        'warm_start': [False],  # For retraining a trained model. Refer -
                                        'class_weight': ['balanced', None] #https://stackoverflow.com/questions/42757892/how-to-use-warm-start/54304493
# For class weight! V.imp!! - https://chrisalbon.com/machine_learning/trees_and_forests/handle_imbalanced_classes_in_random_forests/
                                        }

#SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

    sgd_classifier_params =         {
                                                    'loss': ['modified_huber'],
                                                    'penalty': ['l2', None],
#                                                    'alpha': [0.0001],
#                                                    'l1_ratio': [0.15],
                                                    'fit_intercept': [True],
                                                    'max_iter': [1000],
                                                    'tol': [0.001],
                                                    'shuffle': [True],
                                                    'verbose': [verbose],
                                                    'epsilon': [0.1],
                                                    'n_jobs': [-1],
                                                    'random_state': [seed],
#                                                    'learning_rate': ['optimal', 'adaptive', 'invscaling'],
#                                                    'eta0': [0, 0.1, 0.2],
                                                    'power_t': [0.5],
                                                    'early_stopping': [False],
                                                    'validation_fraction': [0.1],
                                                    'n_iter_no_change': [5],
                                                    'class_weight': [None, 'balanced'],
                                                    'warm_start': [False],
                                                    'average': [False],
                                            }


    one_vs_rest_classifier_params =         {
                                                'n_jobs': [-1]
                                  
                                            }


    model_param_mappings = {
        'RandomForestClassifier': random_forest_params,
        'DecisionTreeClassifier': decision_tree_params,
        'AdaBoostClassifier': adaboost_params,
        'MLPClassifier': mlp_params,
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'LogisticRegression': logistic_regression_params,
        'KNeighborsClassifier': KNeighborsClassifier(),
        'MultinomialNB': MultinomialNB(),
        'XGBClassifier': XGBClassifier(),
        'DummyClassifier': DummyClassifier(),
        'GaussianNB': GaussianNB(),
        'SVC': svc_params,
        'NuSVC': NuSVC(probability = True),
        'BaggingClassifier': bagging_classifier_params,
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB(),
        'ExtraTreesClassifier': extra_trees_classifier_params,
        'GaussianProcessClassifier': GaussianProcessClassifier(),
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(),
        'ExtraTreeClassifier': ExtraTreeClassifier(),
        'LinearSVC': LinearSVC(),
        'NearestCentroid': NearestCentroid(),
        'OneVsOneClassifier': OneVsOneClassifier(MultinomialNB()),
        'OneVsRestClassifier': one_vs_rest_classifier_params,
        'OutputCodeClassifier': OutputCodeClassifier(MultinomialNB()),
        'PassiveAggressiveClassifier': passive_aggressive_classifier_params,
        'Perceptron': Perceptron(),
        'RidgeClassifier': RidgeClassifier(),
        'SGDClassifier': sgd_classifier_params,
        'BayesianGaussianMixture': BayesianGaussianMixture(),
        'GaussianMixture': GaussianMixture()

        }

    return model_param_mappings[func]

##############################################################################################################################

####################<AUC-ROC FOR MULTI-CLASS>########################################
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]
        
        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        
        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    return roc_auc_dict

##############################################################################################################################

##############################<TEST THE MODEL>#################################################
def model_testing(opt, X_test, y_test):
    
    if isinstance(X_test, (np.ndarray)) == False:
        accuracy_test.append('-')
        TP_test.append('-')
        TN_test.append('-')
        FP_test.append('-')
        FN_test.append('-')
        precision_0_test.append('-')
        recall_0_test.append('-')
        f1_score_0_test.append('-')
        precision_1_test.append('-')
        recall_1_test.append('-')
        f1_score_1_test.append('-')
        c_kappa_test.append('-')
        auc_test_0.append('-')
        auc_test_1.append('-')
        LOG_loss_test.append('-')
        
    else:
        print("="*62)
        print('TEST RESULTS FOR ' + (target[-1]) + ', ' + (feature_used[-1]) + ' WITH ' +  (models[-1]) + ' ====>>>>')
        print('DATASET SIZE: {}'.format(dataset_size[-1]))
        print('TRAIN SIZE: {}'.format(X_train.shape[0]))
        print('VAL SIZE: {}'.format(X_val.shape[0]))
        print('TEST SIZE: {}'.format(X_test.shape[0]))
        print('TRAIN POSITIVE SIZE: {}'.format(train_pos[0]))
        print('VAL POSITIVE SIZE: {}'.format(val_pos[0]))
        print('TEST POSITIVE SIZE: {}'.format(test_pos[0]))
        test_predictions = opt.predict(X_test)
        acc = accuracy_score(y_test, test_predictions)
        accuracy_test.append("{0:.3f}".format(acc))
        tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
        TP_test.append(tp)
        TN_test.append(tn)
        FP_test.append(fp)
        FN_test.append(fn)
        class_rep = classification_report(y_test, test_predictions, output_dict = True)
        precision_0_test.append("{0:.3f}".format(class_rep['0.0']['precision']))
        recall_0_test.append("{0:.3f}".format(class_rep['0.0']['recall']))
        f1_score_0_test.append("{0:.3f}".format(class_rep['0.0']['f1-score']))
        precision_1_test.append("{0:.3f}".format(class_rep['1.0']['precision']))
        recall_1_test.append("{0:.3f}".format(class_rep['1.0']['recall']))
        f1_score_1_test.append("{0:.3f}".format(class_rep['1.0']['f1-score']))
        cohen_score = cohen_kappa_score(y_test, test_predictions)
        c_kappa_test.append("{0:.3f}".format(cohen_score))
        print("Accuracy_Test: {:.3%}".format(acc))
        print("Confusion Mat Stats (Test Set):")
        print("TP = {0}, TN = {1}, FP = {2}, FN = {3}".format(tp, tn, fp, fn))
        print("Cohen Score (Test): {:.3f}".format(cohen_score))
        class_rep = classification_report(y_test, test_predictions)
        print("Full Report (Test): ", class_rep)

        auc_outputs = roc_auc_score_multiclass(y_test, test_predictions)
        auc_test_0.append("{0:.3f}".format(auc_outputs[0]))
        auc_test_1.append("{0:.3f}".format(auc_outputs[1]))

        if opt.__class__.__name__ == 'BayesianGaussianMixture' or opt.__class__.__name__ == 'GaussianMixture' or m_name == 'PassiveAggressiveClassifier':
            LOG_loss_test.append('-')
        
        elif hasattr(opt, "predict_proba") and opt.__class__.__name__ != 'BayesianGaussianMixture' and opt.__class__.__name__ != 'GaussianMixture':
            test_predictions_proba = opt.predict_proba(X_test)
            #        test_predictions_1 = test_predictions_proba[:, 0]
            ll = log_loss(y_test, test_predictions_proba)
            LOG_loss_test.append("{0:.3f}".format(ll))
            print("Log Loss (Test): {0:.3f}".format(ll))
        
        else:
            ll = '-'
            LOG_loss_test.append('-')
            print("Log Loss (Test): ", ll)

##############################################################################################################################


##############################<VALIDATE THE MODEL>#################################################
def model_validation(opt, X_val, y_val):
    
    val_predictions = opt.predict(X_val)
    acc_val = accuracy_score(y_val, val_predictions)
    accuracy_val.append("{0:.3f}".format(acc_val))
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, val_predictions).ravel()
    TP_val.append(tp_val)
    TN_val.append(tn_val)
    FP_val.append(fp_val)
    FN_val.append(fn_val)
    #     confusion_mat = confusion_matrix(y, val_predictions)
    class_rep = classification_report(y_val, val_predictions, output_dict = True)
    precision_0_val.append("{0:.3f}".format(class_rep['0.0']['precision']))
    recall_0_val.append("{0:.3f}".format(class_rep['0.0']['recall']))
    f1_score_0_val.append("{0:.3f}".format(class_rep['0.0']['f1-score']))
    precision_1_val.append("{0:.3f}".format(class_rep['1.0']['precision']))
    recall_1_val.append("{0:.3f}".format(class_rep['1.0']['recall']))
    f1_score_1_val.append("{0:.3f}".format(class_rep['1.0']['f1-score']))
    cohen_score_val = cohen_kappa_score(y_val, val_predictions)
    c_kappa_val.append("{0:.3f}".format(cohen_score_val))
    
    auc_outputs = roc_auc_score_multiclass(y_val, val_predictions)
    auc_val_0.append("{0:.3f}".format(auc_outputs[0]))
    auc_val_1.append("{0:.3f}".format(auc_outputs[1]))

    if opt.__class__.__name__ == 'BayesianGaussianMixture' or opt.__class__.__name__ == 'GaussianMixture' or m_name == 'PassiveAggressiveClassifier':
        LOG_loss_val.append('-')
    
    elif hasattr(opt, "predict_proba") and opt.__class__.__name__ != 'BayesianGaussianMixture' and opt.__class__.__name__ != 'GaussianMixture':
        val_predictions_proba = opt.predict_proba(X_val)
        ll = log_loss(y_val, val_predictions_proba)
        LOG_loss_val.append("{0:.3f}".format(ll))
    
    elif not hasattr(opt, "predict_proba"):
        LOG_loss_val.append('-')

    if isinstance(X_test, (np.ndarray)) == False:
        print("="*62)
        print('VAL RESULTS FOR ' + (target[-1]) + ', ' + (feature_used[-1]) + ' WITH ' +  (models[-1]) + ' ====>>>>')
        print('DATASET SIZE: {}'.format(dataset_size[-1]))
        print('TRAIN SIZE: {}'.format(X_train.shape[0]))
        print('VAL SIZE: {}'.format(X_val.shape[0]))
        print('TEST SIZE: {}'.format(0))
        print('TRAIN POSITIVE SIZE: {}'.format(train_pos[0]))
        print('VAL POSITIVE SIZE: {}'.format(val_pos[0]))
        print('TEST POSITIVE SIZE: {}'.format(0))
        print("Accuracy_Val: {:.3%}".format(acc_val))
        print("Confusion Mat Stats (Val Set):")
        print("TP = {0}, TN = {1}, FP = {2}, FN = {3}".format(tp_val, tn_val, fp_val, fn_val))
        print("Cohen Score (Val): {:.3f}".format(cohen_score_val))
        class_rep = classification_report(y_val, val_predictions)
        print("Full Report (Val): ", class_rep)
        print("AUC of Prediction class 1 (Val): {0:.3f}".format(auc_outputs))
        print("AUC of Precision & Recall class 1 (Val): {0:.3f}".format(auc_pre_re))
        print("Log Loss (Val): {0:.3f}".format(ll))

##############################################################################################################################

##############################<DEVELOP THE MODEL>#################################################
def model_development(model, model_type, X_train, X_val, y_train, y_val):
    
    global m_name
    m_name = model.__class__.__name__
    
    model_name = model.__class__.__name__ + '_' + model_type
    models.append(model_name)

    if model_type == 'default':
        default_params = model.get_params()

        for keys in default_params:
            (default_params[keys]) = [default_params[keys]]

        # Run the grid search
        print("=======TRAINING " + model.__class__.__name__ + " MODEL=======")
#        opt = GridSearch(model, param_grid = default_params, seed = seed)
#        opt.fit(X_train, y_train, X_val, y_val, scoring = 'cohen_score', verbose = verbose)
        opt = model
        opt.fit(X_train, y_train)
        save_params.append(str(default_params))

    elif model_type == 'best':

        print("=======TRAINING " + model.__class__.__name__ + " MODEL=======")
        opt = GridSearch(model, param_grid = get_model_params(model.__class__.__name__) , seed = seed)
        opt.fit(X_train, y_train, X_val, y_val, scoring = 'cohen_score', verbose = verbose)
        bes_param = opt.get_best_params()
        
        for keys in bes_param:
            (bes_param[keys]) = [bes_param[keys]]
        
        save_params.append(str(bes_param))

    return opt

##############################################################################################################################

##############################<SAVE RESULTS>#################################################
def save_results(output_path):
    
    #    Creating and saving csv files
    labels = ['Output_Target']
    df_report = pd.DataFrame(target,columns=labels)
    df_report['Input_feature'] = feature_used
    df_report['Models'] = models
    df_report['Hyperparameters_Used'] = save_params
    df_report['Model_RunTime_min'] = model_runtime
    df_report['Dataset_size'] = dataset_size
    df_report['Train_size'] = train_size
    df_report['Val_size'] = val_size
    df_report['Test_size'] = test_size
    df_report['Train_positive_size'] = train_pos
    df_report['Val_positive_size'] = val_pos
    df_report['Test_positive_size'] = test_pos
    df_report['Accuracy_Val'] = accuracy_val
    df_report['True_Positive_Val'] = TP_val
    df_report['True_Negative_Val'] = TN_val
    df_report['False_Positive_Val'] = FP_val
    df_report['False_Negative_Val'] = FN_val
    df_report['Cohen_Score_Val'] = c_kappa_val
    df_report['Precision_0_Val'] = precision_0_val
    df_report['Recall_0_Val'] = recall_0_val
    df_report['F1 Score_0_Val'] = f1_score_0_val
    df_report['AUC_output_0_Val'] = auc_val_0
    df_report['Precision_1_Val'] = precision_1_val
    df_report['Recall_1_Val'] = recall_1_val
    df_report['F1_Score_1_Val'] = f1_score_1_val
    df_report['AUC_output_1_Val'] = auc_val_1
    df_report['Log_Loss_Val'] = LOG_loss_val
    df_report['Accuracy_Test'] = accuracy_test
    df_report['True_Positive_Test'] = TP_test
    df_report['True_Negative_Test'] = TN_test
    df_report['False_Positive_Test'] = FP_test
    df_report['False_Negative_Test'] = FN_test
    df_report['Cohen_Score_Test'] = c_kappa_test
    df_report['Precision_0_Test'] = precision_0_test
    df_report['Recall_0_Test'] = recall_0_test
    df_report['F1_Score_0_Test'] = f1_score_0_test
    df_report['AUC_output_0_Test'] = auc_test_0
    df_report['Precision_1_Test'] = precision_1_test
    df_report['Recall_1_Test'] = recall_1_test
    df_report['F1 Score_1_Test'] = f1_score_1_test
    df_report['AUC_output_1_Test'] = auc_test_1
    df_report['Log_Loss_Test'] = LOG_loss_test
    
    if sys.argv[1] == 'AllModels':
        df_report.to_csv(output_path + '/' + target[-1] + '-' + feature_used[-1] + '-' + sys.argv[1] + '_' + sys.argv[8] + '.csv')
    
    else:
        df_report.to_csv(output_path + '/' + target[-1] + '-' + feature_used[-1] + '-' + models[-1] + '.csv')

##############################################################################################################################

##########################----SAVE THE MODEL----#####################################

def save_model(model_to_save, model_save_path):
    import pickle
    
    model_save_path = model_save_path + '/' + target[-1] + '-' + feature_used[-1] + '-' + models[-1] + '.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_to_save, f)

##############################################################################################################################


##########################----INITIALIZE THE MODEL----#####################################
def model_initialization(model_name, numpy_path_tr, numpy_path_va, numpy_path_te, output_path, _test_size, _val_size, model_type):
    
    #Calculate start time
    start_time = time.time()
    
    # Input - Output split
    X_train, X_val, X_test, y_train, y_val, y_test = get_splitted_data(numpy_path_tr, numpy_path_va, numpy_path_te, _test_size, _val_size)
    
    #Get mapping of model name with its function
    model = get_model_function(model_name)
    
    #Fitting the model
    opt = model_development(model, model_type, X_train, X_val, y_train, y_val)
    
    #Validating the model
    model_validation(opt, X_val, y_val)
    
    #Testing the model
    model_testing(opt, X_test, y_test)
    
    #Calculating end time
    end_minus_start_time = ((time.time() - start_time)/60)
    model_runtime.append("{0:.3f}".format(end_minus_start_time))
    print("MODEL RUNTIME: {:.3f} minutes".format(end_minus_start_time)) #Calculating end time
    print("="*62)
              
    #Saving the model
    if len(sys.argv) < 10 or sys.argv[9] == 'save_model=no':
        print('******************<Model NOT saved>******************')
        print('To save the model, set "save_model=yes" and then give path for saving the model.')
    
    elif sys.argv[9] == 'save_model=yes':
        save_model(opt, sys.argv[10])
        print('******************<Model saved>******************')
    
    #Saving test & val results in csv file
    save_results(output_path)

######################################################################################################

#########################----MAIN FUNCTION BELOW-------###################################
def main():
    
    '''
        model_name, numpy_path, output_path, _test_size, _val_size, model_type
      sys.argv[0] = model_dev_v2.py, sys.argv[1] = model_name, sys.argv[2] = numpy_path,
      sys.argv[3] = output_path, sys.argv[4] = test_size, sys.argv[5] = best/default
    '''
    
    all_models_list = get_model_function('AllModels')
    
    if sys.argv[1] == 'AllModels':
        
        for _model in all_models_list:
            model_initialization(_model, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])

    elif sys.argv[1] in all_models_list:
        model_initialization(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])

    elif sys.argv[1] == 'Get_Models_List':
        print(all_models_list)
        print('No. of models available = {}'.format(len(all_models_list)))

    else:
        print("="*62)
        print('Model not available. Please provide a valid <name_of_Model>, i.e, EXACT string, from the available models.')
        print('To get a list of available models, use the following command:')
        print('python model_dev_v4.py Get_Models_List')
        print("="*62)

if __name__ == "__main__":
     main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################
'''
   ALL CLASSIFIERS HERE SUPPORT <predict> and <predict_proba>
1. Open your command prompt
2. Make sure you are in the correct python env. where sys, os, pandas, numpy, sklearn are installed

3. Command: python model_dev_v2.py <name_of_Model> <numpy_path_of_features> <path_to_save_results> <test_size> <model_type>
            [ Six arguments, in total, after python command: model_dev_v2.py, model_name, numpy_path, output_path, test_size, best/default ]

4. <name_of_Model> :
    a) Available sklearn Models : DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, LogisticRegression,
                                XGBClassifier, MultinomialNB, GaussianNB, KNeighborsClassifier, DummyClassifier, MLPClassifier,
                                SVC, NuSVC, GradientBoostingClassifier, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    b) 15 models available as of now
    c) if <name_of_Model> == 'AllModels' : results will be saved for all models; models will run sequentially and NOT PARALLEL.
    
5. <numpy_path_of_features> :
    a) Eg. path = /Users/path/to/csv/morgan_fp-cytokinetic_bridge.npy
    b) .npy file should contain input (features), and output as the last column.
    c) File name (Eg. morgan_fp-cytokinetic_bridge.npy) should be in the correct format; i.e, name of FP first and then subcell_location seperated by a '-'.
    
6. <path_to_save_results> :
    a) Eg. path = /Users/path/to/save/results
    b) results will be saved in .csv format having the following columns:
    Subcell_Location    Models    Model RunTime (min)    Fingerprint    Dataset size    Train size    Test size    Train positive size    Test positive       size    Accuracy    True Positive    True Negative    False Positive    False Negative    Cohen Score    Precision (0)    Recall (0)    F1 Score (0)    Precision (1)    Recall (1)    F1 Score (1)    AUC output (1)    AUC prec&rec (1)    Log Loss
    
7. <test_size> :
    a) Float number : 0.2 , 0.3, ...
    
8. <model_type> :
    a) if model_type is set to 'default' : Model will fit on train set; prediction will be done on test set. Model will fit on default hyperparameters. Data split only on train and set (no validation dataset). No hyperparameter tuning will happen.
        This method ^ is fast to execute for seeing raw/tentative results.
    b) if model_type is set to 'best' : Train data will split in Kfolds. Hyperparameter tuning will happen via GridSearchCV method. Best model will be saved.
        This method ^ is slow to execute, but will give the best results.
        
9. Eg. of full command :
    a) python model_dev_v2.py DecisionTreeClassifier /abc.npy /Users/results_directory 0.2 best
    b) NOTE:
        - All the 6 arguments are one space seperated
        
    
10. TODO:
    a) Give input for Kfolds.
    b)

'''
