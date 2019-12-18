#######################################################################################
# Author: Srijan Verma                                                              #
# School of Pharmacy                                                                #
# Sirimulla Research Group [http://www.sirimullaresearchgroup.com/]                 #
# The University of Texas at El Paso, TX, USA                                       #
# Last modified: 19/12/2019                                                         #
# Copyright (c) 2019 Srijan Verma and Sirimulla Research Group, under MIT license   #
#######################################################################################
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import time
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
import sklearn

# implemented fingerprints:
# ECFC0 (ecfc0), ECFP0 (ecfp0), MACCS (maccs),
# atom pairs (ap), atom pairs bit vector (apbv), topological torsions (tt)
# hashed atom pairs (hashap), hashed topological torsions (hashtt) --> with 1024 bits
# ECFP4 (ecfp4), ECFP6 (ecfp6), ECFC4 (ecfc4), ECFC6 (ecfc6) --> with 1024 bits
# FCFP4 (fcfp4), FCFP6 (fcfp6), FCFC4 (fcfc4), FCFC6 (fcfc6) --> with 1024 bits
# Avalon (avalon) --> with 1024 bits
# long Avalon (laval) --> with 16384 bits
# long ECFP4 (lecfp4), long ECFP6 (lecfp6), long FCFP4 (lfcfp4), long FCFP6 (lfcfp6) --> with 16384 bits
# RDKit with path length = 5 (rdk5), with path length = 6 (rdk6), with path length = 7 (rdk7)
# 2D pharmacophore (pharm) ?????????????

calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
nbits = 1024
longbits = 16384
seed = 7

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
fpdict['rdkDes'] = lambda m: calc.CalcDescriptors(m)
long_fps = {'laval', 'lecfp4', 'lecfp6', 'lfcfp4', 'lfcfp6'}

def CalculateFP(fp_name, smiles):
    
    m = Chem.MolFromSmiles(smiles)
    return fpdict[fp_name](m)

#####################################################################################################################
def get_fp(fp_name, csv_path, numpy_path):
    
    print("="*62)
    target = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    print('-----------GENERATING {0} FEATURES FOR {1}-----------'.format(fp_name, target))
    
    df = pd.read_csv(csv_path)
    #    df = df[::-1] # REVERSING DF SO THAT POSITIVE COMES AT TOP!!! ALSO, SO THAT POSITIVE DATA DOESN'T GET DROPPED WHILE REMOVING DUPLICATE FPS!!
    df.reset_index(drop=True, inplace=True)
    smiles_list = df['Canonical_Smiles'].tolist()
    output = df['Label'].tolist()
    
    feat_list = fp_name.split('_')
    if len(feat_list) > 1 or fp_name in long_fps:
        _dtype = np.float16
    else:
        _dtype = np.float32

    not_found = []
    features = []
    descriptors = []
    
    for i in tqdm(range(len(smiles_list))):
        feat = ''
        
        for j in range(len(feat_list)):
            
            try:
                fp = CalculateFP(feat_list[j], smiles_list[i])
                if feat_list[j] == 'rdkDes':
                    fp = np.asarray(fp)
                    fp = fp.reshape(1,200)

                else:
                    fp = fp.ToBitString()
                    fp = np.array([(np.fromstring(fp,'u1') - ord('0'))], dtype=_dtype)

            except:
                pass

            if j == 0:
                feat = fp
            
            else:
                feat = np.concatenate((feat, fp), axis=1)
                    
        features.append(feat)
    
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    
    Y = Y.reshape(Y.shape[0],1)
    Y = np.vstack(Y).astype(_dtype)

    X = np.array(features)
    X = X.reshape(X.shape[0], X.shape[2])
    X = np.nan_to_num(X, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
    col_mean = np.nanmean(X, axis = 0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.vstack(X).astype(_dtype)

    print('Output shape: {}'.format(Y.shape))
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input, output and family array
#    final_array = sklearn.utils.shuffle(final_array, random_state=seed)

    print('Final Numpy array shape: {}'.format(final_array.shape))
    print('Type of final array: {}'.format(type(final_array)))
    print('dtype of final array: {}'.format(final_array.dtype))
    np.save(numpy_path + '/' + fp_name + '-' + target +'.npy', np.asarray((final_array), dtype=_dtype))
    print('Input <--> Output')
    #Calculating end time
    end_minus_start_time = ((time.time() - start_time)/60)
    #    model_runtime.append("{0:.3f}".format(end_minus_start_time))
    print("FP RUNTIME: {:.3f} minutes".format(end_minus_start_time)) #Calculating end time
    print('-------------Numpy array saved----------------')
    print("="*62)
#####################################################################################################################

#########################----MAIN FUNCTION BELOW-------###################################

def main():
    
    #Calculate start time
    global start_time
    start_time = time.time()
    
    get_fp(sys.argv[1], sys.argv[2], sys.argv[3])
    all_fp_list = [*fpdict] + ['rdkDes_lecfp6_laval_hashap']
    
#    if sys.argv[1] == 'all_fp':
#        
#        for _fp in all_fp_list:
#            get_fp(_fp, sys.argv[2], sys.argv[3])
#
#    elif sys.argv[1] in all_fp_list:
#        get_fp(sys.argv[1], sys.argv[2], sys.argv[3])
#    
#    elif sys.argv[1] == 'Get_FP_List':
#        print(all_fp_list)
#        print('No. of FPs available = {}'.format(len(all_fp_list)))
#    
#    else:
#        print("="*62)
#        print('FP not available. Please provide a valid <name_of_FP>, i.e, EXACT string, from the available FPs.')
#        print('To get a list of available FPs, use the following command:')
#        print('python generate_fp_v4.py Get_FP_List')
#        print("="*62)

if __name__ == "__main__":
    main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################
'''
    NOTE!!! - Morgan_fp here = ECFP4 (since radius is set to 2)
1. Open your command prompt
2. Make sure you are in the correct python env. where rdkit, pandas, numpy, sklearn are installed
3. Command: python generate_fp_v*.py '<name_of_FP>' '<csv_file_path_having_smiles_and_label>' '<path_to_save_numpy_array>'
4. '<name_of_FP>' :
    a) Type = String ('')
    b) Available FP/other (10) : 'morgan_fp', 'maccs_fp', 'avalon_fp', 'rdk_fp', 'atom_pair_fp', 'torsions_fp', 'topological_fp', 'molecule_descriptors',
                                'ecfp2', 'ecfp6'
    c) if <name_of_FP> == 'all' ; all fp/other will get generated
5. '<csv_file_path_having_smiles_and_label>' :
    a) Type = String ('')
    b) Eg. path = /Users/path/to/csv/abc.csv
    c) csv file should contain atleast 2 columns with names as 'Smiles' and 'Label'; else, error will be shown
6. '<path_to_save_numpy_array>' :
    a) Type = String ('')
    b) Eg. path = /Users/path/where/numpy_array/to/save
    
7. Eg. of full command :
    a) python generate_fp.py 'atom_pair_fp' '../../dataset/cytokinetic_bridge/cytokinetic_bridge_with_smiles.csv' '../../dataset/fp_arrays'
    b) NOTE: All the 3 arguments are one space seperated
    
8. TODO:
    a) 'atom_pair_fp', 'torsions_fp', 'topological_fp' have error in their function; need to fix that.
    b) Add another argument for taking Bit vector size as well.
    c)

'''
