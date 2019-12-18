#######################################################################################
# Author: Srijan Verma                                                              #
# School of Pharmacy                                                                #
# Sirimulla Research Group [http://www.sirimullaresearchgroup.com/]                 #
# The University of Texas at El Paso, TX, USA                                       #
# Last modified: 19/12/2019                                                         #
# Copyright (c) 2019 Srijan Verma and Sirimulla Research Group, under MIT license   #
#######################################################################################
from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
import pandas as pd
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
import os
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import MinMaxScaler

##########################---DOCUMENTATION MORGAN FP GIVEN BELOW---#####################################
##########################< NOTE!!! - Morgan_fp here = ECFP4 > ##########################
########< https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints >########
##############################################################################################

def morgan_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=1024) # default : radius=2,nBits=1024
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated
    
    print('Number of FPs not found: {}'.format(len(not_found)))
    
    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)
    
    print('Output shape: {}'.format(Y.shape))
    
    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))

    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')

##########################---DOCUMENTATION ECFP2 GIVEN BELOW---#####################################
########< https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints >########
##############################################################################################

def ecfp2_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = AllChem.GetMorganFingerprintAsBitVect(m1, radius=1, nBits=1024) # default : radius=2,nBits=1024
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))
    
    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)
    
    print('Output shape: {}'.format(Y.shape))
    
    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')

##########################---DOCUMENTATION ECFP6 GIVEN BELOW---#####################################
########< https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints >########
##############################################################################################

def ecfp6_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = AllChem.GetMorganFingerprintAsBitVect(m1, radius=3, nBits=1024) # default : radius=2,nBits=1024
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))
    
    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)
    
    print('Output shape: {}'.format(Y.shape))
    
    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')


##########################---DOCUMENTATION MACC FP GIVEN BELOW---#####################################
########< https://www.rdkit.org/docs/GettingStartedInPython.html#maccs-keys >########
##############################################################################################


def maccs_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = MACCSkeys.GenMACCSKeys(m1) # Only one option for nBits, which is : nBits=167
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))
    
    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)
    
    print('Output shape: {}'.format(Y.shape))
    
    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')


##########################---DOCUMENTATION AVALON FP GIVEN BELOW---#####################################
########<https://www.rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html>########
##############################################################################################

def avalon_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = pyAvalonTools.GetAvalonFP(m1, nBits=512) #default bits = 512
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))

    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)
    
    print('Output shape: {}'.format(Y.shape))
    
    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')

##########################---DOCUMENTATION RDK FP GIVEN BELOW---#####################################
########< https://www.rdkit.org/docs/source/rdkit.Chem.MolDb.FingerprintUtils.html >########
##############################################################################################

def rdk_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = RDKFingerprint(m1, nBitsPerHash=1) #default bits = 2048
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))

    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)

    print('Output shape: {}'.format(Y.shape))
    
    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')


##########################---DOCUMENTATION ATOM PAIR FP GIVEN BELOW---#####################################
########<https://www.rdkit.org/docs/source/rdkit.Chem.MolDb.FingerprintUtils.html>########
##############################################################################################

def atom_pair_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = Pairs.GetAtomPairFingerprintAsIntVect(m1)
            fp._sumCache = fp.GetTotalVal() #Bit vector here will be huge, which is why taking TotalVal()
#             bits = fp.ToBitString()
#             bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(fp._sumCache)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))

    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)

    print('Output shape: {}'.format(Y.shape))

    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    
    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')



##########################---DOCUMENTATION TORSION FP GIVEN BELOW---#####################################
########<https://www.rdkit.org/docs/source/rdkit.Chem.MolDb.FingerprintUtils.html>########
##############################################################################################

def torsions_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(m1)
            fp._sumCache = fp.GetTotalVal() #Bit vector here will be huge, which is why taking TotalVal()
            #             bits = fp.ToBitString()
            #             bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(fp._sumCache)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))

    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)

    print('Output shape: {}'.format(Y.shape))

    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32

    print('Input shape: {}'.format(X.shape))
    
    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')


##########################---DOCUMENTATION DESCRIPTORS (physical properties) GIVEN BELOW---#####################################
###< https://www.rdkit.org/docs/source/rdkit.ML.Descriptors.MoleculeDescriptors.html >######
#####<https://sourceforge.net/p/rdkit/mailman/message/30087006/>##################
##############################################################################################

def molecule_descriptors(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    descriptors = []
    not_found = []
    index = []
    
    for i in tqdm(range(len(smiles_list))):
        try:
            
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            m1 = Chem.MolFromSmiles(smiles_list[i])
            ds = calc.CalcDescriptors(m1)
            ds = np.asarray(list(ds))
            descriptors.append(ds)
        
        except:
            
            descriptors.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where Descriptor not generated

    print('Number of Descriptors not found: {}'.format(len(not_found)))

    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)

    print('Output shape: {}'.format(Y.shape))

    fp_array = ( np.asarray((descriptors), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where Descriptor not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)                         #Normalize -->( (X-X.min()) / (X.max()-X.min) ) where X.max() & X.min() are taken from within a
                                                        #column and NOT the whole numpy array

    print('Input shape: {}'.format(X.shape))

    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array
    
    ######## Next 3 lines for removing rows, from final_array, where duplicate Descriptors are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate Descriptors: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')

##########################---DOCUMENTATION TOPOLOGICAL FP GIVEN BELOW---#####################################
###< https://www.rdkit.org/docs/GettingStartedInPython.html#topological-fingerprints >######
##############################################################################################

def topological_fp(csv_path, numpy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    output = df['Label'].tolist()
    
    fingerprints = []
    not_found = []
    index = []
    for i in tqdm(range(len(smiles_list))):
        try:
            
            m1 = Chem.MolFromSmiles(smiles_list[i])
            fp = FingerprintMols.FingerprintMol(m1) #default bits = 2048
            bits = fp.ToBitString()
            bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
            fingerprints.append(bits_array)
        
        except:
            
            fingerprints.append(np.nan)
            not_found.append(i)
            pass

    df.drop(not_found, axis=0,inplace=True)             #drop rows where FP not generated

    print('Number of FPs not found: {}'.format(len(not_found)))

    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()                       #Converting 'str' label to numeric label
    Y = labelencoder.fit_transform(df['Label'].values)
    Y = Y.reshape(Y.shape[0],1)

    print('Output shape: {}'.format(Y.shape))

    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)          #drop rows from array where FP not generated
    X = np.vstack(X).astype(np.float32)                 #type convert FROM dtype=object TO dtype=float32

    print('Input shape: {}'.format(X.shape))

    final_array = np.concatenate((X, Y), axis=1)        #Concatenating input and output array

    ######## Next 3 lines for removing rows, from final_array, where duplicate FPs are present
    final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
    _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
    final_array_unique = final_array[unq_row_indices]
    
    print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
    print('Final Numpy array shape: {}'.format(final_array_unique.shape))
    print('Type of final array: {}'.format(type(final_array_unique)))
    subcell_loc = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
    np.save(numpy_path + '/' + sys.argv[1] + '-' + subcell_loc +'.npy', np.asarray((final_array_unique), dtype=np.float32))
    print('-------------Numpy array saved----------------')


#########################----MAIN FUNCTION BELOW-------###################################

def main():
 
    if sys.argv[1] == 'morgan_fp':  #morgan_fp = ecfp_4
        morgan_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'maccs_fp':
        maccs_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'avalon_fp':
        avalon_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'rdk_fp':
        rdk_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'atom_pair_fp':
        atom_pair_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'torsions_fp':
        torsions_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'topological_fp':
        topological_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'molecule_descriptors':
        molecule_descriptors(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'ecfp2_fp':
        ecfp2_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
    
    elif sys.argv[1] == 'ecfp6_fp':
        ecfp6_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

    elif sys.argv[1] == 'all':
        morgan_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path MORGAN_FP = ECFP4 !!!
        maccs_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        avalon_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        rdk_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        atom_pair_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        torsions_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        topological_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        molecule_descriptors(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        ecfp6_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path
        ecfp2_fp(sys.argv[2], sys.argv[3]) #sys.argv[2] = csv_path, sys.argv[3] = numpy_path

if __name__ == "__main__":
     main()


#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################
'''
    NOTE!!! - Morgan_fp here = ECFP4 (since radius is set to 2)
1. Open your command prompt
2. Make sure you are in the correct python env. where rdkit, pandas, numpy, sklearn are installed
3. Command: python generate_fp.py '<name_of_FP>' '<csv_file_path_having_smiles_and_label>' '<path_to_save_numpy_array>'
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
    a) python generate_fp.py 'atom_pair_fp' '../../dataset/cytokinetic_bridge/cyp_3a4.csv' '../../dataset/fp_arrays'
    b) NOTE: All the 3 arguments are one space seperated
    
8. TODO:
    a) 'atom_pair_fp', 'torsions_fp', 'topological_fp' have error in their function; need to fix that.
    b) Add another argument for taking Bit vector size as well.
    c)

'''

