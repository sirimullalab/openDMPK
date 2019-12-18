#######################################################################################
# Author: Srijan Verma                                                              #
# School of Pharmacy                                                                #
# Sirimulla Research Group [http://www.sirimullaresearchgroup.com/]                 #
# The University of Texas at El Paso, TX, USA                                       #
# Last modified: 19/12/2019                                                         #
# Copyright (c) 2019 Srijan Verma and Sirimulla Research Group, under MIT license   #
#######################################################################################
# This program can be used to generate Fingerprints or Molecular Descriptors from RDKit as well as MayaChemTools
import time
import json
import multiprocessing as mp
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
#from rdkit.Chem.Fingerprints import FingerprintMols
import os
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import MinMaxScaler
#from rdkit.Chem.AtomPairs import Pairs
import tempfile
import shutil
#not_found = []
class Features_Generations:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.name = self.csv_path.split('/')[1][:-4]
#        self.not_found = []
    def molecule_descriptors(self):
        """
        Receives the csv file which is used to generate molecular descriptors (200) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Descriptors are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
 
        smiles_list = df['Smiles'].tolist()
        descriptors = []
#        not_found = []        
        print("Extracting smiles features using all " + str(mp.cpu_count()) + " threads...")

        pool = mp.Pool(mp.cpu_count())
        result = pool.map(self.get_features, smiles_list)
        smiles_dict = {}
        for s, f in zip(smiles_list, result):
            smiles_dict[s] = {'features': f}

        print("Extracted for " + str(len(smiles_dict)) + " smiles")
        print("Putting smiles features into data/rdkitdescriptors0.json ...")
        with open(os.path.join("numpy_files_r/"+self.name+'.json'), 'w') as f:
            json.dump(smiles_dict, f)
        pool.close()
        print(len(smiles_dict))
        print("Features extraction completed")
        #print('not_found', self.not_found)
    def get_features(self,smi):
        try:
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            s2=time.time()
            mol = Chem.MolFromSmiles(smi)
#            e2=time.time()
#            t2=(e2-s2)
#            if t2>10.0:
#                return "None"
#            if mol == None:
                #self.not_found.append[smi]
#                return "None"
#            s=time.time()
            ds = calc.CalcDescriptors(mol)
#            e=time.time()
#            t = (e-s)
#            print('time',t)
#            if t>10.0:
#                return "None"
            ds = list(ds)
    #        print('ds',ds)
            return ds
        except:
            #self.not_found.append[smi]
            return "None"
            
fg = Features_Generations(sys.argv[1])
fg.molecule_descriptors()
