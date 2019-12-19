#######################################################################################
# Author: Srijan Verma                                                              #
# School of Pharmacy                                                                #
# Sirimulla Research Group [http://www.sirimullaresearchgroup.com/]                 #
# The University of Texas at El Paso, TX, USA                                       #
# Last modified: 19/12/2019                                                         #
# Copyright (c) 2019 Srijan Verma and Sirimulla Research Group, under MIT license   #
#######################################################################################
import pandas as pd
import math

df = pd.read_csv("data/sult_chembl_bioactivity_18_3_57_35.csv", sep='\t')
new_df = df.loc[:, ["CMPD_CHEMBLID", "COMPOUND_KEY", "MOLWEIGHT", "ALOGP", "PSA", "CANONICAL_SMILES", "STANDARD_TYPE", "STANDARD_VALUE"]]
raw_df = new_df.drop(["STANDARD_TYPE", "STANDARD_VALUE"], axis=1)
raw_df = raw_df.drop_duplicates()

def get_value(x, st="Vmax"):
    df = new_df.loc[new_df.CMPD_CHEMBLID == x["CMPD_CHEMBLID"]]
    value = df.loc[df["STANDARD_TYPE"]==st]["STANDARD_VALUE"].values
    return value[0] if len(value)>0 else None

raw_df["Vmax"] = raw_df.apply(get_value, axis=1)
raw_df["Km"].apply(lambda x: -math.log(x))
raw_df["pKm"] = raw_df["Km"].apply(lambda x: -math.log(x))
raw_df.to_csv("clean_data.csv", index=None)