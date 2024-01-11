import numpy as np
import pandas as pd
from pathlib import Path
import requests
from add_annotations import *
from utils import *
from add_annotations import *
from add_sasa import *
import streamlit as st
import json

UNIPROT_ANNOTATION_COLS = ['disulfide', 'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding',
                           'activeSite',
                           'nucleotideBinding', 'lipidation', 'site', 'transmembrane',
                           'crosslink', 'mutagenesis', 'strand',
                           'helix', 'turn', 'metalBinding', 'repeat', 'topologicalDomain',
                           'caBinding', 'bindingSite', 'region',
                           'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif',
                           'coiledCoil', 'peptide',
                           'transitPeptide', 'glycosylation', 'propeptide', 'disulfideBinary',
                           'intMetBinary', 'intramembraneBinary',
                           'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary',
                           'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
                           'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
                           'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                           'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary',
                           'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
                           'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
                           'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
                           'glycosylationBinary', 'propeptideBinary']
SIMPLE_COLS = ['uniprotID', 'wt', 'pos', 'mut', 'datapoint', 'composition', 'polarity',
       'volume', 'granthamScore', 'domain', 'domStart', 'domEnd', 'distance',
       'intMet', 'naturalVariant', 'activeSite', 'crosslink', 'mutagenesis',
       'strand', 'helix', 'turn', 'region', 'modifiedResidue', 'motif',
       'metalBinding', 'lipidation', 'glycosylation', 'topologicalDomain',
       'nucleotideBinding', 'bindingSite', 'transmembrane', 'transitPeptide',
       'repeat', 'site', 'peptide', 'signalPeptide', 'disulfide', 'coiledCoil',
       'intramembrane', 'zincFinger', 'caBinding', 'propeptide', 'dnaBinding',
        'disulfideBinary', 'intMetBinary', 'intramembraneBinary',
       'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary',
       'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
       'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
       'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
       'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary',
       'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
       'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
       'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
       'glycosylationBinary', 'propeptideBinary']

def addSwissModels(to_swiss, path_to_input_files, path_to_output_files):
    '''
    :param to_swiss:
    :param path_to_input_files:
    :param path_to_output_files:
    :return: swissmodel dataframe with mapped SWISSMODEL information, dataframe that will be sent to modbase.
    '''

    print('\n>>> Proceeding to  SwissModel search...')
    print('------------------------------------\n')

    if len(to_swiss) > 0:
        print('\n>>> Generating SwissModel file...\n')

        to_swiss.reset_index(drop=True, inplace=True)
        to_swiss.fillna(np.NaN)

        swiss_model = pd.read_csv(Path(path_to_input_files / 'swissmodel_structures.txt'),
                                  sep='\t', dtype=str, header=None, skiprows=1,
                                  names=['UniProtKB_ac', 'iso_id', 'uniprot_seq_length', 'uniprot_seq_md5',
                                         'coordinate_id', 'provider', 'from', 'to', 'template', 'qmean_norm', 'seqid',
                                         'url'])
        swiss_model = swiss_model[swiss_model.UniProtKB_ac.isin(to_swiss.uniprotID.to_list())]
        try:
            swiss_model.iso_id = swiss_model.iso_id.astype('str')
        except:
            AttributeError
            swiss_model['iso_id'] = np.NaN
        for ind in swiss_model.index:
            swiss_model.at[ind, 'UniProtKB_ac'] = swiss_model.at[ind, 'UniProtKB_ac'].split('-')[0]
        swiss_model = swiss_model[swiss_model.provider == 'SWISSMODEL']
        print('\n>>> Index File Processed...\n')
        swiss_model = swiss_model[['UniProtKB_ac', 'from', 'to', 'template', 'qmean_norm', 'seqid', 'url']]
        # Sort models on qmean score and identity. Some proteins have more than one models, we will pick one.
        swiss_model = swiss_model.sort_values(by=['UniProtKB_ac', 'qmean_norm', 'seqid'], ascending=False)
        swiss_model.reset_index(inplace=True, drop=True)
        with_swiss_models = to_swiss[to_swiss.uniprotID.isin(swiss_model.UniProtKB_ac.to_list())]
        no_swiss_models = to_swiss[~to_swiss.uniprotID.isin(swiss_model.UniProtKB_ac.to_list())]
        if len(no_swiss_models) == 0:
            no_swiss_models = pd.DataFrame(columns=to_swiss.columns)
        else:
            no_swiss_models.reset_index(drop=True, inplace= True)

        swiss_models_with_data = pd.merge(with_swiss_models, swiss_model, left_on=['uniprotID'],
                                           right_on=['UniProtKB_ac'], how='left')

        swiss_models_with_data = swiss_models_with_data.sort_values(by=['uniprotID', 'wt','pos', 'qmean_norm'],
                                                                    ascending=False)

        swiss_models_with_data['pos'] = swiss_models_with_data['pos'] .apply(lambda x: int(x))
        swiss_models_with_data['from'] = swiss_models_with_data['from'].apply(lambda x: int(x))
        swiss_models_with_data['to'] = swiss_models_with_data['to'] .apply(lambda x: int(x))

        notEncompassed = swiss_models_with_data[((swiss_models_with_data['pos'] > swiss_models_with_data['to']) | (swiss_models_with_data['pos'] < swiss_models_with_data['from']))]
        swiss_models_with_data = swiss_models_with_data[(swiss_models_with_data['pos'] < swiss_models_with_data['to']) & (swiss_models_with_data['pos'] > swiss_models_with_data['from'])]

        notEncompassed = notEncompassed[~notEncompassed.uniprotID.isin(swiss_models_with_data.uniprotID.to_list())]
        swiss_models_with_data = swiss_models_with_data.drop(['UniProtKB_ac', 'seqid'], axis=1)
        swiss_models_with_data = swiss_models_with_data[swiss_models_with_data.url != np.NaN]
        url_nan = swiss_models_with_data[swiss_models_with_data.url == np.NaN]
        url_nan = url_nan.drop(['from', 'qmean_norm', 'template', 'to', 'url'], axis=1)


        no_swiss_models_updated = pd.concat([no_swiss_models, url_nan, notEncompassed])
        if len(swiss_models_with_data)>0:
            for i in swiss_models_with_data.index:
                try:
                    swiss_models_with_data.at[i, 'chain'] = swiss_models_with_data.at[i, 'template'].split('.')[2]
                    swiss_models_with_data.at[i, 'template'] = swiss_models_with_data.at[i, 'template'].split('.')[0]
                except IndexError:
                    swiss_models_with_data.at[i, 'chain'] = np.NaN
                    swiss_models_with_data.at[i, 'template'] = np.NaN

            swiss_models_with_data.chain = swiss_models_with_data.chain.astype('str')
            swiss_models_with_data['qmean_norm'] = swiss_models_with_data.qmean_norm.apply(lambda x: round(float(x), 2))

            no_swiss_models_updated.reset_index(drop = True, inplace=True)
            swiss_models_with_data.reset_index(drop=True, inplace=True)

            existing_free_sasa = list(Path(path_to_output_files / 'freesasa_files').glob("*"))
            existing_free_sasa = [str(i) for i in existing_free_sasa]
            existing_free_sasa = [i.split('/')[-1].split('.')[0] for i in existing_free_sasa]
            print('Beginning SwissModel files download...')
            existing_swiss = list(Path(path_to_output_files / 'swissmodel_structures').glob("*"))
            existing_swiss = [str(i) for i in existing_swiss]
            existing_swiss = ['.'.join(i.split('/')[-1].split('.')[:-1]) for i in existing_swiss]

            for i in swiss_models_with_data.index:
                protein = swiss_models_with_data.at[i, 'uniprotID']
                varPos = swiss_models_with_data.at[i, 'pos']
                wt = swiss_models_with_data.at[i, 'wt']
                template = swiss_models_with_data.at[i, 'template'].split('.')[0]
                qmean_norm = str(round(float(swiss_models_with_data.at[i, 'qmean_norm']), 2))

                swiss_models_with_data.at[i, 'coordVAR'] = np.NaN
                swiss_models_with_data.at[i, 'coordinates'] = np.NaN
                swiss_models_with_data.at[i, 'AAonPDB'] = np.NaN
                varPos = swiss_models_with_data.at[i, 'pos']
                AAonPDB = np.NaN
                coordDict = {}
                if protein + '_' + template + '_' + qmean_norm not in existing_swiss:
                    url = swiss_models_with_data.at[i, 'url'].strip('\"').strip('}').replace('\\', '').strip('\"')
                    req = requests.get(url)
                    name = Path(path_to_output_files / 'swissmodel_structures' / f'{protein}_{template}_{qmean_norm}.txt')
                    print('Downloading for Protein:', protein + ' Model: ' + template)
                    with open(name, 'wb') as f:
                        f.write(req.content)
                else:
                    print(f'Model exists for {protein}.')
                    name = Path(path_to_output_files / 'swissmodel_structures' / f'{protein}_{template}_{qmean_norm}.txt')

                swiss_dp = protein + '_' + template + '_' + qmean_norm
                if swiss_dp not in existing_free_sasa:

                    (run_freesasa(Path(path_to_output_files / 'swissmodel_structures' / f'{swiss_dp}.txt'),
                                  Path(path_to_output_files / 'freesasa_files' / f'{swiss_dp}.txt'), include_hetatms=True,
                                  outdir=None, force_rerun=False, file_type='pdb'))
                  
                filename = Path(path_to_output_files / 'freesasa_files' / f'{swiss_dp}.txt')

                swiss_models_with_data.at[i, 'sasa'] = sasa(protein, varPos, wt, 1, filename, path_to_output_files,
                                                            file_type='pdb')
                with open(name, encoding="utf8") as f:
                    lines = f.readlines()
                    for row in lines:
                        if row[0:4] == 'ATOM' and row[13:15] == 'CA':
                            position = int(row[22:26].strip())
                            chain = row[20:22].strip()
                            aminoacid = threeToOne(row[17:20])
                            coords = [row[31:38].strip(), row[39:46].strip(), row[47:54].strip()]
                            coordDict[position] = coords
                            if int(position) == int(varPos):
                                AAonPDB = aminoacid
                                coordVAR = coords
                        if (row[0:3] == 'TER') or (row[0:3] == 'END'):

                            swiss_models_with_data.loc[i, 'coordinates'] = str(coordDict)
                            swiss_models_with_data.loc[i, 'AAonPDB']     = str(AAonPDB)
                            swiss_models_with_data.loc[i, 'coordVAR']    = str(coordVAR)

                            break

                if swiss_models_with_data.at[i, 'AAonPDB'] == swiss_models_with_data.at[i, 'wt']:
                    swiss_models_with_data.at[i, 'PDB_ALIGN_STATUS'] = 'aligned'
                else:
                    swiss_models_with_data.at[i, 'PDB_ALIGN_STATUS'] = 'notAligned'
            swiss_models_with_data.sort_values(['uniprotID', 'wt', 'pos', 'mut', 'PDB_ALIGN_STATUS', 'qmean_norm'],
                                    ascending=[True, True, True, True, True, False], inplace=True)
            swiss_models_with_data.drop_duplicates(['uniprotID', 'wt', 'pos', 'mut'], keep='first', inplace=True)
            obsolete = swiss_models_with_data[pd.isna(swiss_models_with_data.coordVAR)]
            no_swiss_models_updated = pd.concat([no_swiss_models_updated, obsolete])
            swiss_models_with_data = swiss_models_with_data.fillna(np.NaN)
    else:
        swiss_models_with_data = pd.DataFrame()
        no_swiss_models_updated = pd.DataFrame()

    no_swiss_models_updated = no_swiss_models_updated[SIMPLE_COLS]
    return swiss_models_with_data, no_swiss_models_updated
