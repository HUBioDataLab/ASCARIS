'''
This code merges SwissModel model summary files (metadata) into one file to be used in feature vector creation.
Please run this code in the folder wherein downloaded .tar files are downloaded.
Merged file will be found under extract_swissmodel_structures folder that will be created when this code is run.
'''


import tarfile, glob, os, shutil
import argparse

parser = argparse.ArgumentParser(description='ASCARIS')

parser.add_argument('-folder_name', '--folder_name',
                    help='Enter the directory where meta-data is found.',
                    default=1)

args = parser.parse_args()

meta_data = args.folder_name
def swissmodel_file():
    os.makedirs('input_files/extract_swissmodel_structures/', exist_ok=True)

    all_swissmodel = open('input_files/swissmodel_structures.txt', 'w')
    all_swissmodel.write('UniProtKB_ac	iso_id	uniprot_seq_length	uniprot_seq_md5	coordinate_id	provider	from	to	template	qmeandisco_global	seqid	url')
    all_swissmodel.write('\n')
    for f in glob.glob(f'{meta_data}/*.tar.gz'):
        name = f.split('/')[-1].split('.')[0]
        with tarfile.open(f) as tar:
            tar.extractall(f'input_files/extract_swissmodel_structures/{name}')
            with open(f'input_files/extract_swissmodel_structures/{name}/SWISS-MODEL_Repository/INDEX') as x:
                lines = (x.readlines())[7:]
                for line in lines:
                    all_swissmodel.write(line)
    shutil.rmtree('input_files/extract_swissmodel_structures/')


if __name__ == '__main__':
    swissmodel_file()
