import requests
import numpy as np
import pandas as pd
from utils import *
from pathlib import Path
from bs4 import BeautifulSoup
from add_sasa import *
def addModbaseModels(dataframe, path_to_input_files, path_to_output_files):
    if len(dataframe) != 0:
        # GET MODBASE MODELS
        # Get IDs from data to retrieve only their models from MODBASE
        dataframe.reset_index(inplace=True, drop=True)
        existing_modbase_models = list(Path(path_to_output_files / 'modbase_structures').glob("*"))
        existing_modbase_models = [str(i) for i in existing_modbase_models]
        existing_modbase_models = [i.split('/')[-1].split('.')[0] for i in existing_modbase_models]

        existing_modbase_models_ind = list(Path(path_to_output_files / 'modbase_structures_individual').glob("*"))
        existing_modbase_models_ind = [str(i) for i in existing_modbase_models_ind]
        existing_modbase_models_ind = [i.split('/')[-1].split('.')[0] for i in existing_modbase_models_ind]

        modbase_reduced = pd.DataFrame(columns = ['uniprotID', 'target_begin', 'target_end', 'quality_score',
                                               'model_id', 'coordinates','AAonPDB', 'coordVAR'])
        print('Retrieving ModBase models...\n')
        modbase = pd.DataFrame(
            columns=['uniprotID', 'target_begin', 'target_end', 'quality_score', 'model_id',
                     'coordinates', 'AAonPDB', 'coordVAR'])
        no_modbase = pd.DataFrame(
            columns=['uniprotID', 'target_begin', 'target_end', 'quality_score', 'model_id',
                     'coordinates', 'AAonPDB', 'coordVAR'])
        # Get model files associated with each UniProtID
        existing_free_sasa = list(Path(path_to_output_files / 'freesasa_files').glob("*"))
        existing_free_sasa = [str(i) for i in existing_free_sasa]
        existing_free_sasa = [i.split('/')[-1].split('.')[0] for i in existing_free_sasa]
        keep_cols = dataframe.columns
        for i in dataframe.index:
            coordDict = {}
            protein = dataframe.at[i, 'uniprotID']
            varPos = int(dataframe.at[i, 'pos'])
            wt =  dataframe.at[i, 'wt']
            mut = dataframe.at[i, 'mut']
            datapoint = dataframe.at[i, 'datapoint']
            
            if protein not in existing_modbase_models:
                print('Downloading Modbase models for ', protein)
                url = 'https://salilab.org/modbase/retrieve/modbase/?databaseID=' + protein
                req = requests.get(url)
                name = path_to_output_files / 'modbase_structures' /  f'{protein}.txt'
                with open(name, 'wb') as f:
                    f.write(req.content)
            else:
                print('Model exists for', protein)
                name = Path(path_to_output_files / 'modbase_structures' / f'{protein}.txt')

            with open(name, encoding="utf8") as f:
                a = open(name, 'r').read()
                soup = BeautifulSoup(a, 'lxml')
                if soup.findAll('pdbfile') != []:
                    for pdb in soup.findAll('pdbfile'):
                        model_id = str(pdb.contents[1])[10:-11]
                        if model_id not in existing_modbase_models_ind:
                            with open(path_to_output_files / 'modbase_structures_individual' / f'{model_id}.txt', 'w', encoding="utf8") as individual:
                                individual.write(str('UniProt ID: ' + protein))
                                individual.write('\n')
                                individual.write(str(pdb.contents[3])[10:-11].strip())
                            run_freesasa(
                                Path(path_to_output_files / 'modbase_structures_individual' / f'{model_id.lower()}.txt'),
                                Path(path_to_output_files / 'freesasa_files' / f'{model_id.lower()}.txt'),
                                include_hetatms=True,
                                outdir=None, force_rerun=False, file_type='pdb')
                        filename = Path(path_to_output_files / 'freesasa_files' / f'{model_id.lower()}.txt')
                        sasa_val = sasa(protein, varPos, wt, 1, filename, path_to_output_files, file_type='pdb')
                        with open(path_to_output_files / 'modbase_structures_individual'/ f'{model_id}.txt', encoding="utf8") as m:

                            lines = m.readlines()
                            quality_score = -999
                            for ind_line in lines:
                                if ind_line[0:10] == 'UniProt ID':
                                    uniprot_id = ind_line.split(':')[1].strip()
                                if ind_line[0:23] == 'REMARK 220 TARGET BEGIN':
                                    target_begin = ind_line[40:43].strip()
                                if ind_line[0:21] == 'REMARK 220 TARGET END':
                                    target_end = ind_line[40:43].strip()
                            coordDict, AAonPDB, coordVAR = {},np.NaN,np.NaN
                            if (int(varPos) > int(target_begin)) & (int(varPos) < int(target_end)):
                                coordDict = {}
                                for ind_line in lines:
                                    if ind_line[0:27] == 'REMARK 220 MODPIPE MODEL ID':
                                        model_id = ind_line[40:].strip()
                                    if ind_line[0:15].strip() == 'REMARK 220 MPQS':
                                        quality_score = ind_line[40:].strip()
                                    if ind_line[0:4] == 'ATOM' and ind_line[13:15] == 'CA':
                                        position = int(ind_line[22:26].strip())
                                        chain = ind_line[20:22].strip()
                                        aminoacid = threeToOne(ind_line[17:20])
                                        coords = [ind_line[31:38].strip(), ind_line[39:46].strip(), ind_line[47:54].strip()]
                                        coordDict[position] = coords
                                        if position == int(varPos):
                                            AAonPDB = aminoacid
                                            coordVAR = str(coords)
                                        if ind_line[0:3] == 'TER':
                                            break
                                try:
                                    k = pd.Series(
                                        [uniprot_id, target_begin, target_end,quality_score, model_id, coordDict, AAonPDB, coordVAR, sasa_val])
                                    new_row = {'uniprotID': uniprot_id, 'target_begin': target_begin,
                                               'target_end': target_end, 'quality_score': quality_score,
                                               'model_id': model_id, 'coordinates': coordDict,
                                               'AAonPDB': AAonPDB, 'coordVAR': coordVAR, 'sasa':sasa_val}
                                    modbase_reduced = modbase_reduced.append(new_row, ignore_index=True)
                                    modbase_reduced = modbase_reduced[['uniprotID', 'quality_score', 'model_id', 'coordinates', 'AAonPDB', 'coordVAR', 'sasa']]           
                                    modbase = dataframe.merge(modbase_reduced, on='uniprotID', how='left')
                                    modbase.quality_score = modbase.quality_score.astype(float)
                                    modbase = modbase.sort_values(by=['datapoint', 'quality_score'], ascending=False)
                                    modbase.reset_index(inplace=True, drop=True)
                                    modbase.fillna(np.NaN, inplace=True)
                                    modbase.replace({'\'?\', ': '',
                                                     ', \'?\'': '',
                                                     '(': '', ')': '',
                                                     '[\'?\']': np.NaN,
                                                     '[]': np.NaN,
                                                     'nan-nan': np.NaN,
                                                     '': np.NaN}, inplace=True)
                                except NameError:
                                    print('This file doesnt have Quality Score. Replacer: -999', model_id)
                            else:
                                new_row = {'uniprotID': uniprot_id, 'wt': wt,
                                               'pos': varPos, 'mut': mut, 'datapoint': datapoint }
                                no_modbase = no_modbase.append(new_row, ignore_index=True)
                                
                else:
                    new_row = {'uniprotID': uniprot_id, 'wt': wt,
                                               'pos': varPos, 'mut': mut, 'datapoint': datapoint }
                    no_modbase = no_modbase.append(new_row, ignore_index=True)
                    


    no_modbase_no_Coord = modbase[pd.isna(modbase['coordVAR'])]
    no_modbase = pd.concat([no_modbase, no_modbase_no_Coord])
    modbase = modbase[~pd.isna(modbase['coordVAR'])]
    no_modbase = no_modbase[keep_cols]
    return modbase, no_modbase
