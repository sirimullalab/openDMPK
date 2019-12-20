### *_v<iteration_no>.py: Every py file has gone through many iterations. Use the last iteration file (i.e. the file with the highest iteration no) for analysis

#### 1. Below command for generating features (single or combination). Note: Make sure the SMILES are in their Canonical Format.
a) For single features:
```
python3 generate_fp_v8.py <feature_name> <path_to_csv_file_containing_column_as_Canonical_Smiles_and_Label> <path_to_save_numpy_file>
Eg. command - 
python3 generate_fp_v8.py fcfp4 ../dataset/csv_files/2D6_inh.csv ../dataset/numpy_files
```
</br>
b) For combination of features:
```
Eg. command - 
python3 generate_fp_v8.py fcfp4_lecfp6_rdkDes ../dataset/csv_files/2D6_inh.csv ../dataset/numpy_files
```
</br>
Total available features:</br>
a) Total number of available features: 20
```

['ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6', 'lecfp4', 'lecfp6', 'lfcfp4', 'lfcfp6', 'maccs', 'hashap', 'hashtt', 'avalon', 'laval', 'rdk5', 'rdk6', 'rdk7', 'rdkDes']+ their combination.

```

#### 2. Below command for model training:</br>
a) For single numpy file as input
```
python3 model_dev_v5.py <model_name> <path_to_npy_file> <path_to_save_result> <test_size> <val_size> <model_type> <save_model> <path_to_save_model>
Eg:-
python3 model_dev_v5.py RandomForestClassifier ../dataset/numpy_files/rdk7-3A4_inh.npy ../results_1 0.0 0.20 default save_model=yes ../model_path
```
b) For multiple numpy files as input
```
python3 model_dev_v7.py ExtraTreeClassifier numpy_files/rdkDes_lecfp6_laval_hashap-1A1_ind_tr.npy numpy_files/rdkDes_lecfp6_laval_hashap-1A1_ind_va.npy numpy_files/rdkDes_lecfp6_laval_hashap-1A1_ind_te.npy ../results 0.15 0.15 default save_model=yes ../saved_models
```
Total models:
```
['RandomForestClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier', 'MLPClassifier', 'GradientBoostingClassifier', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'LogisticRegression', 'KNeighborsClassifier', 'MultinomialNB', 'XGBClassifier', 'DummyClassifier', 'GaussianNB', 'SVC', 'NuSVC', 'BaggingClassifier', 'BernoulliNB', 'ComplementNB', 'ExtraTreesClassifier', 'GaussianProcessClassifier', 'HistGradientBoostingClassifier', 'ExtraTreeClassifier', 'LinearSVC', 'NearestCentroid', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'PassiveAggressiveClassifier', 'Perceptron', 'RidgeClassifier', 'SGDClassifier', 'BayesianGaussianMixture', 'GaussianMixture']
```
No. of models available = 33
