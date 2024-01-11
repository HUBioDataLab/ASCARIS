import ssl
import requests as r
from decimal import *
import numpy as np
import pandas as pd
import json
import ast

UNIPROT_ANNOTATION_COLS =  ['disulfide', 'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding',
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

annotation_list = UNIPROT_ANNOTATION_COLS[0:30]

def add_annotations(dataframe):
    print('Downloading UniProt sequence annotations...\n')
    ssl._create_default_https_context = ssl._create_unverified_context

    original_annot_name = ['DISULFID', 'INIT_MET', 'INTRAMEM', 'VARIANT', 'DNA_BIND', 'ACT_SITE', 'NP_BIND', 'LIPID',
                           'SITE', 'TRANSMEM', 'CROSSLNK', 'MUTAGEN', 'STRAND', 'HELIX', 'TURN', 'METAL', 'REPEAT', 'TOPO_DOM',
                           'CA_BIND', 'BINDING', 'REGION', 'SIGNAL', 'MOD_RES', 'ZN_FING', 'MOTIF', 'COILED', 'PEPTIDE',
                           'TRANSIT', 'CARBOHYD', 'PROPEP']

    annotation_list = ['disulfide', 'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                       'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink', 'mutagenesis', 'strand',
                       'helix', 'turn', 'metalBinding', 'repeat', 'topologicalDomain', 'caBinding', 'bindingSite',
                       'region', 'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil', 'peptide',
                       'transitPeptide', 'glycosylation', 'propeptide']

    dataframe = dataframe.reset_index().drop(['index'], axis=1)
    for protein in list(set(dataframe.uniprotID.to_list())):
        print('Retieving annotations for ' + protein)
        uniprot_entry = r.get("http://www.uniprot.org/uniprot/" + protein + ".txt")
        uniprot_entry = uniprot_entry.text.split('\n')
        annot_for_protein = []
        for annotation in original_annot_name:
            for line in uniprot_entry:
                if annotation.strip() in line and line.startswith(
                        'FT') and 'evidence' not in line and 'ECO' not in line and 'note' not in line:
                    annot_for_protein.append(list(filter(None, line.split(' ')))[1:])
        annotations_present = []
        for select in annot_for_protein:
            if select[0] not in annotations_present:
                dataframe.loc[dataframe.uniprotID == protein, select[0]] = str((select[1].replace('..', '-') + '; '))
                annotations_present.append(select[0])
            else:
                dataframe.loc[dataframe.uniprotID == protein, select[0]] += str((select[1].replace('..', '-') + '; '))
        missingAnnotations = list(set(original_annot_name) - set(annotations_present))
        for miss in missingAnnotations:
            dataframe.loc[dataframe.uniprotID == protein, miss] = np.NaN

    for i in range(len(original_annot_name)):
        dataframe = dataframe.rename(columns={original_annot_name[i]: annotation_list[i]})
    # Fix annotation positions
    print('Processing positions...\n')
    for i in dataframe.index:
        all_positions = []
        for annot in annotation_list:
            if (annot != 'disulfide') & (pd.isna(dataframe.at[i, annot]) != True):
                dataframe.at[i, annot] = [x for x in [k.strip() for k in dataframe.at[i, annot].split(';')] if x]
                all_positions.append(dataframe.at[i, annot])
            elif (annot == 'disulfide') & (pd.isna(dataframe.at[i, annot]) != True):
                dataframe.at[i, annot] = dataframe.at[i, annot].split(';')
                dataframe.at[i, annot] = [i.split('-') for i in dataframe.at[i, annot]]
                dataframe.at[i, annot] = [e for v in  dataframe.at[i, annot] for e in v]
                dataframe.at[i, annot] = [i for i in dataframe.at[i, annot] if i != ' ']
                all_positions.append(dataframe.at[i, annot])
            dataframe.at[i, annot] = str(dataframe.at[i, annot])
        all_positions = [item for sublist in all_positions for item in sublist]
        updated_allPos = []
        for pos in all_positions:
            if '-' in pos:
                first = pos.split('-')[0]
                second = pos.split('-')[1]
                newPos = list(range(int(first), int(second)+1))
                updated_allPos += newPos
            else:
                updated_allPos.append(int(pos))
        updated_allPos.append(dataframe.at[i, 'pos'])
        updated_allPos.append(dataframe.at[i, 'domEnd'])
        updated_allPos.append(dataframe.at[i, 'domStart'])
        updated_allPos = [int(i) for i in updated_allPos]
        dataframe.loc[i, 'POSITIONS'] = str(list(set(updated_allPos)))

    # Add binary annotations
    print('Adding binary annotations...\n')
    for i in dataframe.index:
        for k in annotation_list:  # get the positions of each attribute as a list
            txt = k + 'Binary'
            dataframe.at[i, txt] = np.NaN
            try:
                for positions in dataframe.at[i, k].split(','):
                    position = positions.strip('[').strip(']').replace("'", "")
                    if (position != np.NaN) and (position != '') and ('-' not in position) and (int(
                            dataframe.at[i, 'pos']) == int(position)):
                        dataframe.at[i, txt] = '1'
                        break
                    elif (position != np.NaN) and (position != '') and ('-' not in position) and (int(
                            dataframe.at[i, 'pos']) != int(position)):
                        dataframe.at[i, txt] = '0'
                    elif (position != np.NaN) and (position != '') and ('-' in position):
                        if int(position.split('-')[0]) < int(dataframe.at[i, 'pos']) < int(position.split('-')[1]):
                            dataframe.at[i, txt] = '1'
                            break
                        else:
                            dataframe.at[i, txt] = '0'
            except:
                ValueError
    # Final corrections
    dataframe = dataframe.replace({'[\'?\']': np.NaN})
    dataframe = dataframe.replace({'[]': np.NaN})
    dataframe = dataframe.replace({'': np.NaN})
    dataframe = dataframe.fillna(np.NaN)
    return dataframe

def changeUPtoPDB(dataframe):
    for i in dataframe.index:
        for col in annotation_list:
            newList = []
            if dataframe.at[i, col] != np.NaN:
                if type(dataframe.at[i, col]) == str:
                    list_v = dataframe.at[i, col][1:-1].split(',')
                    positionList = [i.strip().strip('\'') for i in list_v]
                elif type(dataframe.at[i, col]) == list:
                    positionList = dataframe.at[i, col]
                else:
                    positionList = []
                for position in positionList:
                    if '-' in position:
                        all_annots = list(range(int(position.split('-')[0]), int(position.split('-')[1])+1))
                        for annot in all_annots:
                            try:
                                newList.append(ast.literal_eval(dataframe.at[i, 'MATCHDICT'])[str(annot)])
                            except KeyError:
                                pass
                            except TypeError:
                                pass
                    else:
                        try:
                            newList.append(ast.literal_eval(dataframe.at[i, 'MATCHDICT'])[str(position)])
                        except KeyError:
                            pass
                        except TypeError:
                            pass
            dataframe.loc[i, col] = str(newList)
    return dataframe


def changeUPtoModels(dataframe):
    dataframe.fillna(np.NaN, inplace=True)
    for i in dataframe.index:
        for col in annotation_list:
            newList = []
            if (dataframe.at[i, col] != np.NaN) or (type(dataframe.at[i, col]) != 'float'):
                if (type(dataframe.at[i, col]) == str) and  (str(dataframe.at[i, col]) != 'nan') :
                    list_v = dataframe.at[i, col][1:-1].split(',')
                    positionList = [i.strip().strip('\'') for i in list_v]
                elif type(dataframe.at[i, col]) == list:
                    positionList = dataframe.at[i, col]
                else:
                    positionList = []

                if positionList != []:
                    for position in positionList:
                        if '-' in position:
                            all_annots = list(range(int(position.split('-')[0]), int(position.split('-')[1])+1))
                            newList += all_annots
                        else:
                            newList.append(str(position))
                            pass
                else:
                    all_annots = np.NaN
            else:
                all_annots = np.NaN
            newList = [str(i) for i in newList]

            dataframe.loc[i, col] = str(newList)

    return dataframe


def isZeroDistance(data):
    data.fillna(np.NaN, inplace=True)
    for i in data.index:

        for col in UNIPROT_ANNOTATION_COLS[0:30]:
            if data.at[i, col] != np.NaN:
                if type(data.at[i, col]) != 'dict':
                    annotList = ast.literal_eval(data.at[i, col])
                else:
                    annotList = data.at[i, col]
                annotList = [int(i.strip()) for i in annotList if i != 'null']
                if int(data.at[i, 'pos']) in annotList:
                    data.at[i, col] = 'hit'
    return data
    
