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
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsClassifier
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
import pickle

model_runtime = []
test_size = []
feature_size = []
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
auc_out_test = []
auc_prec_rec_test = []
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

##############################<SPLIT DATA INTO TRAIN, VAL AND TEST>##############################
def get_splitted_data(numpy_path):
    
    global X_test, y_test, arr
    
    feature_and_target = (os.path.split(numpy_path)[1][0:len(os.path.split(numpy_path)[1])-4])
    feature_used.append(feature_and_target.split('-')[0])
    target.append(feature_and_target.split('-')[1])

    arr = np.load(numpy_path)
    X_test = arr[:,0:(arr.shape[1]-1)]
    y_test = arr[:,(arr.shape[1]-1)]
    y_test = y_test.reshape((arr.shape[0]),)
    
    test_size.append(X_test.shape[0])
    feature_size.append(X_test.shape[1])
    test_pos.append(np.count_nonzero(y_test))

    return ( X_test, y_test )

##############################################################################################################################

##############################<TEST THE MODEL>#################################################
def model_testing(opt, X_test, y_test):
    
    print("="*62)
    print('TEST RESULTS FOR ' + (target[-1]) + ', ' + (feature_used[-1]) + ' WITH ' +  (models[-1]) + ' ====>>>>')
    print('TEST SIZE: {}'.format(X_test.shape[0]))
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
    
    test_predictions_proba = opt.predict_proba(X_test)
    test_predictions_1 = test_predictions_proba[:, 1]
    auc_outputs = roc_auc_score(y_test, test_predictions_1)
    print("AUC of Prediction class 1 (Test): {0:.3f}".format(auc_outputs))
    auc_out_test.append("{0:.3f}".format(auc_outputs))
    precision, recall, thresholds = precision_recall_curve(y_test, test_predictions_1)
    auc_pre_re = auc(recall, precision)
    print("AUC of Precision & Recall class 1 (Test): {0:.3f}".format(auc_pre_re))
    auc_prec_rec_test.append("{0:.3f}".format(auc_pre_re))
    ll = log_loss(y_test, test_predictions_proba)
    LOG_loss_test.append("{0:.3f}".format(ll))
    print("Log Loss (Test): {0:.3f}".format(ll))

##############################################################################################################################

##############################<SAVE RESULTS>#################################################
def save_results(output_path):
    
    #    Creating and saving csv files
    labels = ['Output_Target']
    df_report = pd.DataFrame(target,columns=labels)
    df_report['Input_feature'] = feature_used
    df_report['Models'] = models
    df_report['Model_RunTime_min'] = model_runtime
    df_report['Test_size'] = test_size
    df_report['Feature_size'] = feature_size
    df_report['Test_positive_size'] = test_pos
    df_report['Accuracy_Test'] = accuracy_test
    df_report['True_Positive_Test'] = TP_test
    df_report['True_Negative_Test'] = TN_test
    df_report['False_Positive_Test'] = FP_test
    df_report['False_Negative_Test'] = FN_test
    df_report['Cohen_Score_Test'] = c_kappa_test
    df_report['Precision_0_Test'] = precision_0_test
    df_report['Recall_0_Test'] = recall_0_test
    df_report['F1_Score_0_Test'] = f1_score_0_test
    df_report['Precision_1_Test'] = precision_1_test
    df_report['Recall_1_Test'] = recall_1_test
    df_report['F1 Score_1_Test'] = f1_score_1_test
    df_report['AUC_output_1_Test'] = auc_out_test
    df_report['AUC_prec_rec_1_Test'] = auc_prec_rec_test
    df_report['Log_Loss_Test'] = LOG_loss_test
    
    df_report.to_csv(output_path + '/' + target[-1] + '-' + feature_used[-1] + '-' + models[-1] + '.csv')

##############################################################################################################################

##########################----LOAD THE MODEL----#####################################
def load_model(model_file):
    
    target_feature_model = ( os.path.split(model_file)[1][0:len(os.path.split(model_file)[1])-4] )
    models.append(target_feature_model.split('-')[2])
    
    with open(model_file, 'rb') as file:
        opt = pickle.load(file)

    return opt

######################################################################################################

##########################----INITIALIZE THE MODEL----#####################################
def model_initialization(numpy_path, model_file, output_path):
 
    #Calculate start time
    start_time = time.time()

    #Input - Output split of prediction data
    X_test, y_test = get_splitted_data(numpy_path)
    
    #Load the model
    opt = load_model(model_file)
 
    #Testing the model
    model_testing(opt, X_test, y_test)
    
    #Calculating end time
    end_minus_start_time = ((time.time() - start_time)/60)
    model_runtime.append("{0:.3f}".format(end_minus_start_time))
    print("MODEL RUNTIME: {:.3f} minutes".format(end_minus_start_time)) #Calculating end time
    print("="*62)
    
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
    model_initialization(sys.argv[1], sys.argv[2], sys.argv[3])

#    all_models_list = get_model_function('AllModels')
#
#    if sys.argv[1] == 'AllModels':
#
#        for _model in all_models_list:
#            model_initialization(_model, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
#
#    elif sys.argv[1] in all_models_list:
#        model_initialization(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
#
#    elif sys.argv[1] == 'Get_Models_List':
#        print(all_models_list)
#        print('No. of models available = {}'.format(len(all_models_list)))
#
#    else:
#        print("="*62)
#        print('Model not available. Please provide a valid <name_of_Model>, i.e, EXACT string, from the available models.')
#        print('To get a list of available models, use the following command:')
#        print('python model_dev_v4.py Get_Models_List')
#        print("="*62)

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
