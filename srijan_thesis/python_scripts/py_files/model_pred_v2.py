#######################################################################################
# Author: Srijan Verma                                                              #
# School of Pharmacy                                                                #
# Sirimulla Research Group [http://www.sirimullaresearchgroup.com/]                 #
# The University of Texas at El Paso, TX, USA                                       #
# Last modified: 19/12/2019                                                         #
# Copyright (c) 2019 Srijan Verma and Sirimulla Research Group, under MIT license   #
#######################################################################################
import pickle
from glob import glob
import pandas as pd
import os
import time
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import rdMolDescriptors


##############################<TEST THE MODEL>#################################################
def model_testing(opt, X_test):
    
    
    test_predictions = opt.predict(X_test)[0]
    
    if not hasattr(opt, "predict_proba"):
        return test_predictions, ''
    
    elif mod == 'PassiveAggressiveClassifier' or mod == 'Perceptron' or mod == 'SGDClassifier' or mod == 'LinearSVC' or mod == 'RidgeClassifier':
       return test_predictions, ''
       
    else:
       
       test_predictions_prob = opt.predict_proba(X_test)
       
       if test_predictions == 0.0:
           return test_predictions , round(test_predictions_prob[0][0], 2)
       
       else:
           return test_predictions, round(test_predictions_prob[0][1], 2)

##############################################################################################################################

##############################<SAVE RESULTS>#################################################
def save_results(df_report, save_pred_path, csv_smi_path):
    
    csv_file_name = os.path.splitext(os.path.basename(csv_smi_path))[0]
    df_report.to_csv(save_pred_path + '/' + csv_file_name + '-prediction_results.csv')

##############################################################################################################################

##########################----LOAD THE MODEL----#####################################
def load_model(model_file):

    with open(model_file, 'rb') as file:
        opt = pickle.load(file)

    return opt

######################################################################################################

nbits = 1024
longbits = 16384

# dictionary
fpdict = {}
fpdict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpdict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpdict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpdict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
#fpdict['ecfc0'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
#fpdict['ecfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
#fpdict['ecfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
#fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
fpdict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpdict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpdict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
#fpdict['fcfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
#fpdict['fcfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
#fpdict['fcfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
fpdict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpdict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpdict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpdict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpdict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
#fpdict['ap'] = lambda m: Pairs.GetAtomPairFingerprint(m)
#fpdict['tt'] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
fpdict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpdict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpdict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpdict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)

long_fps = {'laval', 'lecfp4', 'lecfp6', 'lfcfp4', 'lfcfp6'}

######################################################################################################

##########################----CALCULATE FEATURE----#####################################

def CalculateFP(fp_name, smiles):
    
    m = Chem.MolFromSmiles(smiles)
    return fpdict[fp_name](m)

##########################################################################################

##########################----FEATURE GENERATION----#####################################

def get_features(smi, mod_file):
    
    global fp_name, mod, target
    target_fp_mod = os.path.splitext(os.path.basename(mod_file))[0]
    target = target_fp_mod.split('-')[0]
    fp_name = target_fp_mod.split('-')[1]
    mod = target_fp_mod.split('-')[2].split('_')[0]
    
    target_list.append(target)
    
    if fp_name in long_fps:
        _dtype = np.float16
    
    else:
        _dtype = np.float32
    
    try:

        fp = CalculateFP(fp_name, smi)
        bits = fp.ToBitString()
        bits = [bits]
        X = np.array([(np.fromstring(fp,'u1') - ord('0')) for fp in (bits)], dtype=_dtype)

    except:
        
        X = np.nan
        pass
#        print('Feature could not be generated for ' + smi + '. Please give a valid Smiles')

    return X

######################################################################################################

##########################----INITIALIZE THE MODEL----#####################################

def model_initialization(csv_smi_path, mod_file_path, save_pred_path):
    
    print("="*62)
    
    global target_list
    df = pd.read_csv(csv_smi_path)
    smiles_list = df['Smiles'].tolist()
    
    all_mod_files = sorted(glob(mod_file_path + '/*.pkl'))

    #Input - Output split of prediction data
    for i in range(len(smiles_list)):
        target_list = []
        test_pred_list = []

        for pkl in all_mod_files:
            
            X_test = get_features(smiles_list[i], pkl)

            #Load the model
            opt = load_model(pkl)
            
            if X_test is np.nan:
                test_pred = 'Feature_not_generated_Check_Smiles_again'
                test_pred_list.append(test_pred)
         
            else:
                #Testing the model
                test_pred, test_pred_proba = model_testing(opt, X_test)
                
                test_pred_list.append(str(test_pred) + ' [Probability: ' + str(test_pred_proba) + ']')
    
        if X_test is np.nan:
            print("---------------<Features for Smiles at position " + str(i+1) + " not generated. Check Smiles again>---------------")
    
        else:
            print("---------------<Features and predictions for Smiles at position " + str(i+1) + " generated>---------------")
        
        test_pred_list = [test_pred_list]
        
        if i == 0:
            df_report = pd.DataFrame(columns=target_list)
            
            df_report = df_report.append(pd.DataFrame(test_pred_list, columns=target_list),ignore_index=True)

        else:
            df_report = df_report.append(pd.DataFrame(test_pred_list, columns=target_list),ignore_index=True)

    df_report.insert(loc=0, column='Smiles', value=smiles_list)

    #Saving test & val results in csv file
    save_results(df_report, save_pred_path, csv_smi_path)


    print("-----------------------<Predictions for " + str(i+1) + " Smiles saved>-----------------------")
    print("="*62)

######################################################################################################

#########################----MAIN FUNCTION BELOW-------###################################
def main():
    
    model_initialization(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == "__main__":
     main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################

