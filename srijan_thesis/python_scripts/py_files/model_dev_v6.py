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
train_active = []
train_inactive = []
train_indeterminate = []
val_active = []
val_inactive = []
val_indeterminate = []
test_active = []
test_inactive = []
test_indeterminate = []
models = []
feature_used = []
target = []
accuracy_val = []

TP_val_0 = []
TN_val_0 = []
FP_val_0 = []
FN_val_0 = []
TP_val_1 = []
TN_val_1 = []
FP_val_1 = []
FN_val_1 = []
TP_val_2 = []
TN_val_2 = []
FP_val_2 = []
FN_val_2 = []

TP_test_0 = []
TN_test_0 = []
FP_test_0 = []
FN_test_0 = []
TP_test_1 = []
TN_test_1 = []
FP_test_1 = []
FN_test_1 = []
TP_test_2 = []
TN_test_2 = []
FP_test_2 = []
FN_test_2 = []

accuracy_test = []
c_kappa_val = []
LOG_loss_val = []
precision_0_val = []
recall_0_val = []
f1_score_0_val = []
precision_1_val = []
recall_1_val = []
f1_score_1_val = []
precision_2_val = []
recall_2_val = []
f1_score_2_val = []
auc_val_0 = []
auc_val_1 = []
auc_val_2 = []
#auc_prec_rec_val = []
c_kappa_test = []
LOG_loss_test = []
precision_0_test = []
recall_0_test = []
f1_score_0_test = []
precision_1_test = []
recall_1_test = []
f1_score_1_test = []
precision_2_test = []
recall_2_test = []
f1_score_2_test = []
auc_test_0 = []
auc_test_1 = []
auc_test_2 = []
#auc_prec_rec_test = []
arr = []
X_train = []
X_val = []
X_test = []
y_train = []
y_val = []
y_test = []
save_params = []

seed = 7
verbose = 1
num_classes = 3

##############################<SPLIT DATA INTO TRAIN, VAL AND TEST>##############################
def get_splitted_data(numpy_path, _test_size, _val_size):
    
    global X_train, X_val, X_test, y_train, y_val, y_test, arr
    
    feature_and_target = (os.path.split(numpy_path)[1][0:len(os.path.split(numpy_path)[1])-4])
    feature_used.append(feature_and_target.split('-')[0])
    target.append(feature_and_target.split('-')[1])

    arr = np.load(numpy_path)
    X = arr[:,0:(arr.shape[1]-1)]
    Y = arr[:,(arr.shape[1]-1)]
    Y = Y.reshape((arr.shape[0]),)
    _val_size = float(_val_size)
    _test_size = float(_test_size)
    
    if _test_size == 0.0:
        X_test = 0
        y_test = 0
        test_active.append(0)
        test_inactive.append(0)
        test_indeterminate.append(0)
        test_size.append(0)
        X_train, X_val, y_train, y_val = model_selection.train_test_split(X, Y, test_size=(round((_val_size/(1-_test_size)), 3)), random_state=seed, stratify=Y)
    
    else:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=_test_size, random_state=seed, stratify=Y)
        test_active.append(np.count_nonzero(y_test == 0))
        test_inactive.append(np.count_nonzero(y_test == 1))
        test_indeterminate.append(np.count_nonzero(y_test == 2))
        test_size.append(X_test.shape[0])

        X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=(round((_val_size/(1-_test_size)), 3)), random_state=seed, stratify=y_train)
    
    train_active.append(np.count_nonzero(y_train == 0))
    train_inactive.append(np.count_nonzero(y_train == 1))
    train_indeterminate.append(np.count_nonzero(y_train == 2))

    val_active.append(np.count_nonzero(y_val == 0))
    val_inactive.append(np.count_nonzero(y_val == 1))
    val_indeterminate.append(np.count_nonzero(y_val == 2))

    dataset_size.append(arr.shape[0])
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
                            'n_estimators': [i for i in range(50, 1000, 50)],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None] + [i for i in range(1, 30, 4)],
                            'min_samples_split': np.linspace(0.2, 2.0, 10, endpoint=True),
                            'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
                            'min_weight_fraction_leaf': [0.0],
                            'max_features': ['auto'],
                            'max_leaf_nodes': [None],
                            'min_impurity_decrease': [0.0],
                            # 'min_impurity_split' :
                            'bootstrap':[True],
                            'oob_score': [False],   #  When data too less, then set this to true : reference -
                            'n_jobs': [-1],   #  https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710
                            'random_state': [seed],
                            #  'verbose':[0],
                            'warm_start': [False],  # For retraining a trained model. Refer -
                            'class_weight': ['balanced'] #https://stackoverflow.com/questions/42757892/how-to-use-warm-start/54304493
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
        x = [64, 128, 256]
        hl = []

        for i in range(1, len(x)):
            hl.extend([p for p in itertools.product(x, repeat=i+1)])

        return hl

    hidden_layer_sizes = get_hidden_layers()


    mlp_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['adam'],
                    'alpha': 10.0 ** -np.arange(1, 7),
                    'batch_size': ['auto'],
                    'learning_rate': ['constant'],
                    'learning_rate_init':[0.001, 0.01, 0.0001],
                    'power_t':[0.5],
                    'max_iter':[200],
                    'shuffle': [True],
                    'random_state':[seed],
                    'tol':[0.0001],
                    'verbose':[False],
                    'warm_start':[False],
                    'momentum':[0.9],
                    'nesterovs_momentum':[True],
                    'early_stopping':[True],
                    'validation_fraction':[0.1],
                    'beta_1':[0.9],
                    'beta_2':[0.999],
                    'epsilon':[0.00000001],
                    'n_iter_no_change':[10]
                }# batch_size = min(200, n_samples)

    svc_params  =	{
							'C': [0.001, 0.01, 0.1, 1, 10],
							'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
							'degree': [0, 1, 2, 3, 4, 5],
							'gamma':[0.0001 ,0.001, 0.01, 0.1, 1],
							'coef0': [0.0], 
							'shrinking': [True],
							'probability':[True], 
							'tol': [0.001], 
							'verbose': [False], 
							'max_iter': [-1],
							'decision_function_shape': ['ovr'], # one-vs-one (‘ovo’) is always used as multi-class strategy 
							'random_state':[seed]				#Refer - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
					}
    
    model_param_mappings = {
        'RandomForestClassifier': random_forest_params,
        'DecisionTreeClassifier': decision_tree_params,
        'AdaBoostClassifier': adaboost_params,
        'MLPClassifier': mlp_params,
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'MultinomialNB': MultinomialNB(),
        'XGBClassifier': XGBClassifier(),
        'DummyClassifier': DummyClassifier(),
        'GaussianNB': GaussianNB(),
        'SVC': svc_params,
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
        'OutputCodeClassifier': OutputCodeClassifier(MultinomialNB()),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
        'Perceptron': Perceptron(),
        'RidgeClassifier': RidgeClassifier(),
        'SGDClassifier': SGDClassifier(),
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
        auc_out_test.append('-')
        auc_prec_rec_test.append('-')
        LOG_loss_test.append('-')
        
    else:
        print("="*62)
        print('TEST RESULTS FOR ' + (target[-1]) + ', ' + (feature_used[-1]) + ' WITH ' +  (models[-1]) + ' ====>>>>')
        print('DATASET SIZE: {}'.format(arr.shape[0]))
        print('TRAIN SIZE: {}'.format(X_train.shape[0]))
        print('VAL SIZE: {}'.format(X_val.shape[0]))
        print('TEST SIZE: {}'.format(X_test.shape[0]))
        print('TRAIN ACTIVE SIZE: {}'.format(train_active[0]))
        print('VAL ACTIVE SIZE: {}'.format(val_active[0]))
        print('TEST ACTIVE SIZE: {}'.format(test_active[0]))
        test_predictions = opt.predict(X_test)
        acc = accuracy_score(y_test, test_predictions)
        accuracy_test.append("{0:.3f}".format(acc))

        cm1 = confusion_matrix(y_test, test_predictions)

        tp_test = np.diag(cm1)
        TP_test_0.append(tp_test[0])
        TP_test_1.append(tp_test[1])
        TP_test_2.append(tp_test[2])
            
        FalsePositive = []
        for i in range(num_classes):
            FalsePositive.append(sum(cm1[:,i]) - cm1[i,i])
        FP_test_0.append(FalsePositive[0])
        FP_test_1.append(FalsePositive[1])
        FP_test_2.append(FalsePositive[2])

        FalseNegative = []
        for i in range(num_classes):
            FalseNegative.append(sum(cm1[i,:]) - cm1[i,i])
        FN_test_0.append(FalseNegative[0])
        FN_test_1.append(FalseNegative[1])
        FN_test_2.append(FalseNegative[2])

        TrueNegative = []
        for i in range(num_classes):
            temp = np.delete(cm1, i, 0)   # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TrueNegative.append(sum(sum(temp)))
        TN_test_0.append(TrueNegative[0])
        TN_test_1.append(TrueNegative[1])
        TN_test_2.append(TrueNegative[2])

        class_rep = classification_report(y_test, test_predictions, output_dict = True)
        precision_0_test.append("{0:.3f}".format(class_rep['0.0']['precision']))
        recall_0_test.append("{0:.3f}".format(class_rep['0.0']['recall']))
        f1_score_0_test.append("{0:.3f}".format(class_rep['0.0']['f1-score']))
        precision_1_test.append("{0:.3f}".format(class_rep['1.0']['precision']))
        recall_1_test.append("{0:.3f}".format(class_rep['1.0']['recall']))
        f1_score_1_test.append("{0:.3f}".format(class_rep['1.0']['f1-score']))
        precision_2_test.append("{0:.3f}".format(class_rep['2.0']['precision']))
        recall_2_test.append("{0:.3f}".format(class_rep['2.0']['recall']))
        f1_score_2_test.append("{0:.3f}".format(class_rep['2.0']['f1-score']))
        cohen_score = cohen_kappa_score(y_test, test_predictions)
        c_kappa_test.append("{0:.3f}".format(cohen_score))
        print("Accuracy_Test: {:.3%}".format(acc))
        print("Confusion Mat Stats (Test Set):")
#        print("TP = {0}, TN = {1}, FP = {2}, FN = {3}".format(tp, tn, fp, fn))
        print("Cohen Score (Test): {:.3f}".format(cohen_score))
        class_rep = classification_report(y_test, test_predictions)
        print("Full Report (Test): ", class_rep)

        auc_outputs = roc_auc_score_multiclass(y_test, test_predictions)
        auc_test_0.append("{0:.3f}".format(auc_outputs[0]))
        auc_test_1.append("{0:.3f}".format(auc_outputs[1]))
        auc_test_2.append("{0:.3f}".format(auc_outputs[2]))

        if opt.__class__.__name__ == 'BayesianGaussianMixture' or opt.__class__.__name__ == 'GaussianMixture':
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
    cm1 = confusion_matrix(y_val, val_predictions)
    
    tp_val = np.diag(cm1)
    TP_val_0.append(tp_val[0])
    TP_val_1.append(tp_val[1])
    TP_val_2.append(tp_val[2])
    
    FalsePositive = []
    for i in range(num_classes):
        FalsePositive.append(sum(cm1[:,i]) - cm1[i,i])
    FP_val_0.append(FalsePositive[0])
    FP_val_1.append(FalsePositive[1])
    FP_val_2.append(FalsePositive[2])
    
    FalseNegative = []
    for i in range(num_classes):
        FalseNegative.append(sum(cm1[i,:]) - cm1[i,i])
    FN_val_0.append(FalseNegative[0])
    FN_val_1.append(FalseNegative[1])
    FN_val_2.append(FalseNegative[2])

    TrueNegative = []
    for i in range(num_classes):
        temp = np.delete(cm1, i, 0)   # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TrueNegative.append(sum(sum(temp)))
    TN_val_0.append(TrueNegative[0])
    TN_val_1.append(TrueNegative[1])
    TN_val_2.append(TrueNegative[2])

#
#    TN_val.append(tn_val)
#    FP_val.append(fp_val)
#    FN_val.append(fn_val)
    #     confusion_mat = confusion_matrix(y, val_predictions)
    class_rep = classification_report(y_val, val_predictions, output_dict = True)
    precision_0_val.append("{0:.3f}".format(class_rep['0.0']['precision']))
    recall_0_val.append("{0:.3f}".format(class_rep['0.0']['recall']))
    f1_score_0_val.append("{0:.3f}".format(class_rep['0.0']['f1-score']))
    precision_1_val.append("{0:.3f}".format(class_rep['1.0']['precision']))
    recall_1_val.append("{0:.3f}".format(class_rep['1.0']['recall']))
    f1_score_1_val.append("{0:.3f}".format(class_rep['1.0']['f1-score']))
    precision_2_val.append("{0:.3f}".format(class_rep['2.0']['precision']))
    recall_2_val.append("{0:.3f}".format(class_rep['2.0']['recall']))
    f1_score_2_val.append("{0:.3f}".format(class_rep['2.0']['f1-score']))
    cohen_score_val = cohen_kappa_score(y_val, val_predictions)
    c_kappa_val.append("{0:.3f}".format(cohen_score_val))

#    val_predictions_1 = val_predictions_proba[:, 0]
    auc_outputs = roc_auc_score_multiclass(y_val, val_predictions)
    auc_val_0.append("{0:.3f}".format(auc_outputs[0]))
    auc_val_1.append("{0:.3f}".format(auc_outputs[1]))
    auc_val_2.append("{0:.3f}".format(auc_outputs[2]))
#    auc_out_val.append("{0:.3f}".format(auc_outputs))
#    precision, recall, thresholds = precision_recall_curve(y_val, val_predictions_1)
#    auc_pre_re = auc(recall, precision)
#    auc_prec_rec_val.append("{0:.3f}".format(auc_pre_re))
    if opt.__class__.__name__ == 'BayesianGaussianMixture' or opt.__class__.__name__ == 'GaussianMixture':
        LOG_loss_val.append('-')

    elif hasattr(opt, "predict_proba") and opt.__class__.__name__ != 'BayesianGaussianMixture' and opt.__class__.__name__ != 'GaussianMixture':
        val_predictions_proba = opt.predict_proba(X_val)
        ll = log_loss(y_val, val_predictions_proba)
        LOG_loss_val.append("{0:.3f}".format(ll))

    elif not hasattr(opt, "predict_proba"):
        LOG_loss_val.append('-')

    elif isinstance(X_test, (np.ndarray)) == False:
        print("="*62)
        print('VAL RESULTS FOR ' + (target[-1]) + ', ' + (feature_used[-1]) + ' WITH ' +  (models[-1]) + ' ====>>>>')
        print('DATASET SIZE: {}'.format(arr.shape[0]))
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
    df_report['Train_active_size'] = train_active
    df_report['Val_active_size'] = val_active
    df_report['Test_active_size'] = test_active
    df_report['Train_inactive_size'] = train_inactive
    df_report['Val_inactive_size'] = val_inactive
    df_report['Test_inactive_size'] = test_inactive
    df_report['Train_indeterminate_size'] = train_indeterminate
    df_report['Val_indeterminate_size'] = val_indeterminate
    df_report['Test_indeterminate_size'] = test_indeterminate

    df_report['Accuracy_Val'] = accuracy_val
    df_report['TP_val_0'] = TP_val_0
    df_report['TN_val_0'] = TN_val_0
    df_report['FP_val_0'] = FP_val_0
    df_report['FN_val_0'] = FN_val_0
    df_report['TP_val_1'] = TP_val_1
    df_report['TN_val_1'] = TN_val_1
    df_report['FP_val_1'] = FP_val_1
    df_report['FN_val_1'] = FN_val_1
    df_report['TP_val_2'] = TP_val_2
    df_report['TN_val_2'] = TN_val_2
    df_report['FP_val_2'] = FP_val_2
    df_report['FN_val_2'] = FN_val_2

    df_report['Cohen_Score_Val'] = c_kappa_val
    df_report['Precision_0_Val'] = precision_0_val
    df_report['Recall_0_Val'] = recall_0_val
    df_report['F1 Score_0_Val'] = f1_score_0_val
    df_report['Precision_1_Val'] = precision_1_val
    df_report['Recall_1_Val'] = recall_1_val
    df_report['F1_Score_1_Val'] = f1_score_1_val
    df_report['Precision_2_Val'] = precision_2_val
    df_report['Recall_2_Val'] = recall_2_val
    df_report['F1_Score_2_Val'] = f1_score_2_val
    df_report['AUC_output_0_Val'] = auc_val_0
    df_report['AUC_output_1_Val'] = auc_val_1
    df_report['AUC_output_2_Val'] = auc_val_2
   
    df_report['Log_Loss_Val'] = LOG_loss_val
    df_report['Accuracy_Test'] = accuracy_test
    df_report['TP_test_0'] = TP_test_0
    df_report['TN_test_0'] = TN_test_0
    df_report['FP_test_0'] = FP_test_0
    df_report['FN_test_0'] = FN_test_0
    df_report['TP_test_1'] = TP_test_1
    df_report['TN_test_1'] = TN_test_1
    df_report['FP_test_1'] = FP_test_1
    df_report['FN_test_1'] = FN_test_1
    df_report['TP_test_2'] = TP_test_2
    df_report['TN_test_2'] = TN_test_2
    df_report['FP_test_2'] = FP_test_2
    df_report['FN_test_2'] = FN_test_2
    df_report['Cohen_Score_Test'] = c_kappa_test
    df_report['Precision_0_Test'] = precision_0_test
    df_report['Recall_0_Test'] = recall_0_test
    df_report['F1_Score_0_Test'] = f1_score_0_test
    df_report['Precision_1_Test'] = precision_1_test
    df_report['Recall_1_Test'] = recall_1_test
    df_report['F1 Score_1_Test'] = f1_score_1_test
    df_report['Precision_2_Test'] = precision_2_test
    df_report['Recall_2_Test'] = recall_2_test
    df_report['F1 Score_2_Test'] = f1_score_2_test
    df_report['AUC_output_0_Test'] = auc_test_0
    df_report['AUC_output_1_Test'] = auc_test_1
    df_report['AUC_output_2_Test'] = auc_test_2
    
    df_report['Log_Loss_Test'] = LOG_loss_test
    
    if sys.argv[1] == 'AllModels':
        df_report.to_csv(output_path + '/' + target[-1] + '-' + feature_used[-1] + '-' + sys.argv[1] + '_' + sys.argv[6] + '.csv')
    
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
def model_initialization(model_name, numpy_path, output_path, _test_size, _val_size, model_type):
    
    #Calculate start time
    start_time = time.time()
    
    # Input - Output split
    X_train, X_val, X_test, y_train, y_val, y_test = get_splitted_data(numpy_path, _test_size, _val_size)
    
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
    if len(sys.argv) < 8 or sys.argv[7] == 'save_model=no':
        print('******************<Model NOT saved>******************')
        print('To save the model, set "save_model=yes" and then give path for saving the model.')
    
    elif sys.argv[7] == 'save_model=yes':
        save_model(opt, sys.argv[8])
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
            model_initialization(_model, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

    elif sys.argv[1] in all_models_list:
        model_initialization(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

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
