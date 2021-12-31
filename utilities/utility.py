# Utilitiy script for feature generation
# Md Mahmudulla Hassan
# Jason Sanchez
# The University of Texas at El Paso
# Last Modified: 12/28/2021

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from rdkit.Chem import AllChem
from scipy import sparse
import tempfile
import shutil
import subprocess

class ModelGenerator:
    def __init__(self):
        self.tpatf_len=2692
    
    def smiletoSDF(self, smile):
        # Try to get the rdkit mol
        mol = Chem.MolFromSmiles(smile)
    
        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)
        mol.SetProp("smiles", smile)

        w = Chem.SDWriter(os.path.join(self.temp_dir, r"temp.sdf"))
        w.write(mol)
        w.flush()
    
    def smiletoTPATF(self, smile):
        features = []
        script_path = os.path.abspath(r"../mayachemtools/bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl")

        # Generate the sdf file
        self.smiletoSDF(smile)

        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): 
            print("SDF not found")
            return None

        command = r"perl " + script_path + r" -r " + os.path.join(self.temp_dir, r"temp") + r" --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(self.temp_dir, r"temp.sdf")
        
        subprocess.run(command.split(" "), capture_output=True)
        
        with open(os.path.join(self.temp_dir, r"temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]

        return features

    def get_matricies(self, inputpath, smilecolumnname, labelcolumnname, excel=True):



        input_split=os.path.split(inputpath)
        input_head=input_split[0]
        input_tail=input_split[1]
        input_tail_no_ext=os.path.splitext(input_tail)[0]

        matrix_path=os.path.join(input_head, "matricies")

        if not os.path.exists(matrix_path):
            os.mkdir(matrix_path)

        output_x=os.path.join(matrix_path, input_tail_no_ext +"_x.npz")  
        output_y=os.path.join(matrix_path, input_tail_no_ext +"_y.npz")  


        if not os.path.isfile(output_x):
            print("Developing features for "+ input_tail_no_ext)
            
            if (excel):
                train_df = pd.read_excel(inputpath)
            else:
                train_df=pd.read_csv(inputpath)
            
            train_x_npy = np.zeros((len(train_df),self.tpatf_len))

            with tempfile.TemporaryDirectory() as self.temp_dir:
                for i in tqdm(range(len(train_df))):
                    try:
                        train_x_npy[i, :]=self.smiletoTPATF(train_df[smilecolumnname][i])
                    except Exception as e:
                        print("Something went wrong with smile {}".format(i))
                        print(e)
                        train_x_npy[i, :]=None

                
            train_y_npy=np.array(train_df[labelcolumnname])
            
            train_x_npz=sparse.csr_matrix(train_x_npy)  
            train_y_npz=sparse.csr_matrix(train_y_npy)
            
            sparse.save_npz(output_x, train_x_npz)
            sparse.save_npz(output_y, train_y_npz)
            
        else:
            print("Existing data files found. No data is being generated")
    

    def train_model(self, xpath, ypath):

        outname=os.path.basename(xpath[0:-6])

        model_file_path=r"../models/"+outname+".rf"

        if not os.path.exists(model_file_path):
        
            train_x=sparse.load_npz(xpath)
            train_y=sparse.load_npz(ypath)
            
            if not os.path.isfile(model_file_path):
                clf = RandomForestClassifier(class_weight="balanced", verbose=2)
                param_grid = {"n_estimators": [i for i in range(100, 1001, 100)]}
                grid_clf = GridSearchCV(estimator=clf, cv=5, param_grid=param_grid, n_jobs=-1)
                
                train_x=train_x.toarray()
                train_y=train_y.toarray().T.flatten()
                
                keep_train=~np.isnan(train_x).any(axis=1)

                train_x=train_x[keep_train]
                train_y=train_y[keep_train]
                
                grid_clf.fit(train_x, train_y)

                self.curr_model = grid_clf.best_estimator_
                
                print("TRAINING PERFORMANCE")
                y_pred = self.curr_model.predict(train_x)
                print(classification_report(y_true=train_y, y_pred=y_pred))


                print("Saving Model as: {}".format(outname))
                joblib.dump(self.curr_model, model_file_path)
        
        else:
            print("Existing model found. No model developed")

    def load_model(self, modelpath):
        self.curr_model=joblib.load(modelpath)

    def test_model(self, xpath, ypath, modelpath=None):
        if modelpath is not None:
            try:
                self.load_model(modelpath)
            except:
                print("Model Not Found")

        test_x=sparse.load_npz(xpath)
        test_y=sparse.load_npz(ypath)
        
        test_x=test_x.toarray()
        test_y=test_y.toarray().T.flatten()
        
        keep_test=~np.isnan(test_x).any(axis=1)

        test_x=test_x[keep_test]
        test_y=test_y[keep_test]
        
        print("TESTING PERFORMANCE")
        y_pred = self.curr_model.predict(test_x)
        print(classification_report(y_true=test_y, y_pred=y_pred))

    def train_test(self, path_train, smile_column_train, label_column_train, 
    path_test, smile_column_test, label_column_test,
    excel=True):

        #Creating the training matricies
        self.get_matricies(path_train, smile_column_train, label_column_train, excel=excel)

        #Training the model
        input_split=os.path.split(path_train)
        input_head=input_split[0]
        input_tail=input_split[1]
        input_tail_no_ext=os.path.splitext(input_tail)[0]

        matrix_path=os.path.join(input_head, "matricies")

        output_x=os.path.join(matrix_path, input_tail_no_ext +"_x.npz")  
        output_y=os.path.join(matrix_path, input_tail_no_ext +"_y.npz")  

        self.train_model(output_x, output_y)

        #Creating the testing matricies
        self.get_matricies(path_test, smile_column_test, label_column_test, excel=excel)

        #Loading the model
        outname=os.path.basename(output_x[0:-6])
        model_file_path=r"../models/"+outname+".rf"
        self.load_model(model_file_path)

        #Testing the model
        input_split=os.path.split(path_test)
        input_head=input_split[0]
        input_tail=input_split[1]
        input_tail_no_ext=os.path.splitext(input_tail)[0]

        matrix_path=os.path.join(input_head, "matricies")

        output_x=os.path.join(matrix_path, input_tail_no_ext +"_x.npz")  
        output_y=os.path.join(matrix_path, input_tail_no_ext +"_y.npz") 

        self.test_model(output_x, output_y) 




