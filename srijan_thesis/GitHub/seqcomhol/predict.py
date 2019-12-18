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
import os
import time
import sys
import numpy as np
from rdkit import Chem
import argparse
import pprint
from config import LocInfo_dict, fpFunc_dict, long_fps, fps_to_generate, dict_all, ModFileName_LoadedModel_dict
import multiprocessing as mp

##############################<TEST THE MODEL>#################################################
def model_testing(opt, X_test, mod):
    
    test_predictions = opt.predict(X_test)[0]
    
#    if mod == 'PassiveAggressiveClassifier' or mod == 'SGDClassifier' or mod == 'LinearSVC':
    if not hasattr(opt, "predict_proba"):
        if test_predictions == 0.0:
            return 'inactive', None
        
        else:
            return 'active', None

    else:
       test_predictions_prob = opt.predict_proba(X_test)
       
       if test_predictions == 0.0:
           return 'inactive' , round(test_predictions_prob[0][0], 2)
       
       else:
           return 'active', round(test_predictions_prob[0][1], 2)

##############################################################################################################################

##########################----LOAD THE MODEL----#####################################
def load_model(model_file):
    
    target_fp_mod = os.path.splitext(os.path.basename(model_file))[0][0:-8]
    
    with open(model_file, 'rb') as file:
        opt = pickle.load(file)
    
    ModFileName_LoadedModel_dict[target_fp_mod] = opt

######################################################################################################

##########################----CALCULATE FEATURE----#####################################

def CalculateFP(fp_name, smiles):
    
    m = Chem.MolFromSmiles(smiles)
    return fpFunc_dict[fp_name](m)

##########################################################################################

########################----MULTI-PROCESS FOR PREDICTION----######################################
def multi_process(loaded_model, arr):

    output_dict = {}
    target = loaded_model.split('-')[0]
    fp_name = loaded_model.split('-')[1]
    mod = loaded_model.split('-')[2]

    if arr is None:
        output_dict[target] = {
                                'prediction': None,
                                'probability': None,
                                'model': None,
                                'no_of_actives': None,
                                'feature': None,
                                'accuracy_test': None
                                }
    else:
    
        #Get the model
        opt = ModFileName_LoadedModel_dict[loaded_model]

        #Get predictions
        test_pred, test_pred_proba = model_testing(opt, arr, mod)

        output_dict[target] = {
                                'prediction': test_pred,
                                'probability': test_pred_proba,
                                'model': mod,
                                'no_of_actives': LocInfo_dict[0][target]['actives'],
                                'feature': fp_name,
                                'accuracy_test': LocInfo_dict[0][target]['accuracy_test']
                                }

    return output_dict

##############################----MULTI-PROCESS_FPs----##################################
def multi_process_fp(_smi, _fp):
    
    fpName_array_dict = {}

    if _fp in long_fps:
        _dtype = np.float16
        
    else:
        _dtype = np.float32

    try:
        
        fp = CalculateFP(_fp, _smi)
        bits = fp.ToBitString()
        bits = [bits]
        X = np.array([(np.fromstring(fp,'u1') - ord('0')) for fp in (bits)], dtype=_dtype)
        
    except:
        X = None
        pass

    fpName_array_dict[_fp] = X
    return fpName_array_dict

##########################----INITIALIZE THE MODEL----#####################################

def model_initialization(smi_list):
    
    all_mod_files = sorted(glob('saved_models/*.pkl'))
    
    #Loop over and Load all models in memory and store in a dict--> key = model_file_name ; value = model
    for i in range(len(all_mod_files)):
        load_model(all_mod_files[i])

    #Use all CORES
    pool = mp.Pool(mp.cpu_count())

    #Loop over list of SMILES
    for j in range(len(smi_list)):
        
        #If empty string, then save None
        if smi_list[j] == '':
            dict_all[smi_list[j]] = None
            continue

        #Multi processing for generation of 11 features (for j th smiles)
        final_result_fp = {}
        result_fp = pool.starmap(multi_process_fp, [(smi_list[j], k) for k in fps_to_generate])

        #final_result_fp is a dict, for j th smiles, having 16 FPs--> key = fp_name ; value = array
        for e in result_fp:
            final_result_fp.update(e)

        #If all features are not none, for j th smiles, then predict
        if any(x is not None for x in final_result_fp.values()):
            result = pool.starmap(multi_process, [(k, final_result_fp[k.split('-')[1]]) for k in list(ModFileName_LoadedModel_dict.keys())])
            
            final_result = {}
            for d in result:
                final_result.update(d)
            
            dict_all[smi_list[j]] = final_result

        #If all features are None, then save None for j th smiles
        else:
            dict_all[smi_list[j]] = None
            continue

    pprint.pprint(dict_all)
#    return dict_all

######################################################################################################

#########################----MAIN FUNCTION BELOW-------###################################
def main():
    
    #Calculate start time
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', action='store', dest='smiles', required=False, type=str, help='SMILES list')
    args = parser.parse_args()

    if not (args.smiles):
        parser.error('No input is given, add --smiles')

    if args.smiles:
        smi_list = [sm for sm in args.smiles.split(',')]
        model_initialization(smi_list)

    #Calculating end time
    end_minus_start_time = ((time.time() - start_time))
    print("RUNTIME: {:.3f} seconds".format(end_minus_start_time)) #Calculating end time

if __name__ == "__main__":
     main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################

'''
    1. Example command: python3 predict.py --smiles CCCCO,CCCCCC
    2. Takes one or a list of arguments (SMILES string/SMILES list) [seperated by a comma] and outputs a prediction dictionary for all the subcellular locations
'''
