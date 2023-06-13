import pdb_featureVector
import alphafold_featureVector
import argparse

parser = argparse.ArgumentParser(description='ASCARIS')

parser.add_argument('-s', '--source_option',
                    help='Selection of input structure data.\n 1: PDB Structures (default), 2: AlphaFold Structures',
                    default=1)
parser.add_argument('-i', '--input_datapoint',
                    help='Input file or query datapoint\n Option 1: Comma-separated list of idenfiers (UniProt ID-wt residue-position-mutated residue (e.g. Q9Y4W6-N-432-T or Q9Y4W6-N-432-T, Q9Y4W6-N-432-T)) \n Option 2: Enter comma-separated file path')

parser.add_argument('-impute', '--imputation_state', default='True',
                    help='Whether resulting feature vector should be imputed or not. Default True.')

args = parser.parse_args()

input_set = args.input_datapoint
mode = args.input_option
impute = args.imputation_state

def run_featureVector(input_set, mode, impute):
    print('*****************************************')
    print('Feature vector generation is in progress. \nPlease check log file for updates..')
    print('*****************************************')
    mode = int(mode)
    if mode == 1:
        pdb_featureVector.pdb(input_set, mode, impute)
    elif mode == 2:
        alphafold_featureVector.alphafold(input_set, mode, impute)

if __name__ == '__main__':
    run_featureVector(input_set, mode, impute)


