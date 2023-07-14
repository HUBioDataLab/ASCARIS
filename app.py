import streamlit as st
import pandas as pd 
from os import path
import sys
sys.path.append('code/')
#sys.path.append('ASCARIS/code/') 
import pdb_featureVector
import alphafold_featureVector
import argparse

source = st.selectbox('Source',[1,2])
impute = st.selectbox('Impute',[True, False])
input_data = st.text_input('Input')




#sys.path.append(path.abspath('../code/'))
parser = argparse.ArgumentParser(description='ASCARIS')

parser.add_argument('-s', '--source_option',
                    help='Selection of input structure data.\n 1: PDB Structures (default), 2: AlphaFold Structures',
                    default=1)
parser.add_argument('-i', '--input_datapoint',
                    help='Input file or query datapoint\n Option 1: Comma-separated list of idenfiers (UniProt ID-wt residue-position-mutated residue (e.g. Q9Y4W6-N-432-T or Q9Y4W6-N-432-T, Q9Y4W6-N-432-T)) \n Option 2: Enter comma-separated file path')

parser.add_argument('-impute', '--imputation_state', default='True',
                    help='Whether resulting feature vector should be imputed or not. Default True.')

args = parser.parse_args()

input_set = input_data
mode = source
impute = impute

print('*****************************************')
print('Feature vector generation is in progress. \nPlease check log file for updates..')
print('*****************************************')
mode = int(mode)
if mode == 1:
    pdb_featureVector.pdb(input_set, mode, impute)
    st.write('Process DONE')
elif mode == 2:
    alphafold_featureVector.alphafold(input_set, mode, impute)