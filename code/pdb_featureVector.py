# IMPORT NECESSARY MODULES AND LIBRARIES
from timeit import default_timer as timer
import xml.etree.ElementTree as ET
from collections import Counter
from bs4 import BeautifulSoup
from io import StringIO
from decimal import *
import pandas as pd
import requests
import os.path as op
import subprocess
import shutil
import ssbio.utils
import warnings
import sys
import pathlib
from pathlib import Path
import os, glob
import math
import ssbio
import ssl
import numpy as np
from Bio.Align import substitution_matrices
from Bio.PDB.Polypeptide import *
from Bio.PDB import PDBList
from Bio import Align
from Bio import SeqIO
from Bio.PDB import *
warnings.filterwarnings("ignore")
start = timer()

# FUNCTIONS
from calc_pc_property import *
from add_domains import *
from retrieveUniprotSequences import *
from add_annotations import *
from add_sequence import *
from add_structure import *
from manage_files import *
from add_sasa import *
from standard import *
from add_interface_pos import *
from standard import *
from utils import *
from pdbMapping import *
from uniprotSequenceMatch import uniprotSequenceMatch
from process_input import clean_data
from urllib.error import HTTPError
from swissModelAdd import *
from modbaseModelAdd import *
import streamlit as st


def pdb(input_set, mode, impute):

    # Fill empty dataframes with SIMPLE_COLS
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

    UNIPROT_ANNOTATION_COLS = SIMPLE_COLS[-60:]


    path_to_input_files, path_to_output_files, path_to_domains, fisher_path, path_to_interfaces, buffer = manage_files(mode)
    out_path = path_to_output_files / 'log.txt'
    #sys.stdout = open(out_path, 'w')
    data = clean_data(input_set)
    if len(data) == 0:
        st.write('Feature vectore generation terminated. Please enter a query or check your input format.')
    else:
        data = add_uniprot_sequence(data)
        match = data[(data.wt_sequence_match == 'm')]
        org_len = len(match)
        iso = data[(data.wt_sequence_match == 'i')]
        noMatch = data[(data.wt_sequence_match != 'm') & (data.wt_sequence_match != 'i')]

        if len(noMatch) == len(data) :
            st.write('Aminoacid at the position could not be mapped to canonical or isoform sequence. Please check the input amino acid.')
        elif len(noMatch) > 0:
            st.write(
                f'{len(noMatch)} of {len(data)} datapoints has not been mapped to any sequence. These datapoints are omitted.')
        if len(iso) > 0:
            st.write(f'{len(iso)} of {len(data)} datapoints has been mapped to isoform sequences. These datapoints are omitted.')
        if len(match) == 0:
            st.write('Feature generation terminated due to failed mapping of input amino acid to UniProt sequence.')
        else:
            st.write(f'{len(match)} of {len(data)} datapoints has been mapped to canonical sequences. Proceeding with these datapoins.')
            if (len(iso) != 0) | (len(noMatch) != 0):
                st.write('Omitted datapoints are:', noMatch.datapoint.to_list() + iso.datapoint.to_list())
            st.write('\n')
            st.write('Check log file for updates.')
            
            data = match[['uniprotID', 'wt', 'pos', 'mut', 'datapoint']]
            print('>> Feature vector generation started...\n')
            print('\n>> Creating directories...')
            print('\n>> Adding physicochemical properties...\n')
            data = add_physicochemical(data)
            print('\n>> Adding domains\n')
            data = add_domains(data, path_to_domains)
            print('\n>> Adding sequence annotations...\n')
            data = add_annotations(data)
            print('\n>> Retrieving PDB structure information...\n')
            pdb_info = addPDBinfo(data, path_to_output_files)
            if len(pdb_info) != 0:
                data = pd.merge(data, pdb_info, on='uniprotID', how='left')
                # Spare datapoint if there is no associated PDB.
                no_pdb = data[data.pdbID.isna()].drop_duplicates()
                pdb = data[~data.pdbID.isna()].drop_duplicates()
                # Spare datapoint if associated PDB does not cover mutated area.
                pdb.pos = pdb.pos.apply(lambda x:int(x))
                pdb.start = pdb.start.apply(lambda x: int(x))
                pdb.end = pdb.end.apply(lambda x: int(x))
                no_pdb_add = pdb[~((pdb.pos > pdb.start) & (pdb.pos < pdb.end))]
    
                pdb = pdb[(pdb.pos > pdb.start) & (pdb.pos < pdb.end)] # do not change order
    
                pdb.reset_index(drop=True, inplace=True)
                # Delete spared datapoint from no_pdb list if it has any other PDB that spans the mutated area.
                no_pdb_add = no_pdb_add[~no_pdb_add.datapoint.isin(pdb.datapoint.to_list())]
                # Final collection of datapoints without PDB associaton.
                no_pdb = pd.concat([no_pdb, no_pdb_add])
                no_pdb = no_pdb[SIMPLE_COLS]
                no_pdb = no_pdb.drop_duplicates()
    
                pdb = pdb.sort_values(['uniprotID', 'resolution'], axis=0, ascending=True)
                pdb.reset_index(drop=True, inplace=True)
                pdb.fillna(np.NaN, inplace=True)
                # Get position mapping from added structures
                print('\n>> Adding structure residue positions...\n')
                if len(pdb) > 0: # there are mapped structures, and some of them span the mutated area.
                    pdb.replace({'[]': np.NaN, 'nan-nan': np.NaN, '': np.NaN}, inplace=True)
                    pdb = pdbMapping(pdb, Path(path_to_output_files / 'pdb_structures'))
                    pdb.reset_index(drop=True, inplace=True)
                    pdb = pdb.fillna(np.NaN)
                    no_pdb_add_ = pdb[pdb.AAonPDB.isna()]
                    no_pdb_add = pdb[pdb.MATCHDICT.isna()]
                    no_pdb = pd.concat([no_pdb_add_, no_pdb, no_pdb_add])
                    no_pdb.reset_index(inplace=True, drop=True)
                    pdb = pdb[~(pdb.MATCHDICT.isna())]
                    pdb = pdb[~(pdb.AAonPDB.isna())]
                    if len(pdb) > 0:
                        print('\n>> Mapping to PDB residues...\n')
                        pdb = changeUPtoPDB(pdb)
                        pdb.reset_index(drop=True, inplace=True)
                        print('\n>> Calculating 3D distances for PDB structures...\n')
                        pdb = isZeroDistance(pdb)
                        pdb = processFile(pdb, path_to_output_files)
                        pdb = match3D(pdb)
                        pdb = selectMaxAnnot(pdb)
                        pdb = pdb.sort_values(by=['datapoint', 'resolution', 'annotTotal'], ascending=[True, True, True])
                        pdb = pdb.drop_duplicates(['datapoint'])
                        pdb.replace({'[]': np.NaN, 'hit':0.0}, inplace=True)
                        print('\n>> PDB matching is completed...\n')
                    else:
                        # There was no residue match in the associated PDB. So we cannot use PDB data.
                        pdb = pdb[SIMPLE_COLS]
                        print('\n>>> No PDB structure could be matched.')
    
                else:
                    pdb = pdb[SIMPLE_COLS]
                    print('\n>>> No PDB structure could be matched.')
    
    
            else:
                pdb = pd.DataFrame(columns = SIMPLE_COLS)
                print('\n>>> No PDB structure could be matched.')
                no_pdb = data.copy()
            no_pdb = no_pdb[SIMPLE_COLS]
    
            print(
                'PDB phase is finished...\nPDB structures are found for %d of %d.\n%d of %d failed to match with PDB structure.\n'
                % (len(pdb.drop_duplicates(['datapoint'])), len(data.drop_duplicates(['datapoint'])),
                   len(no_pdb.drop_duplicates(['datapoint'])), len(data.drop_duplicates(['datapoint']))))
    
    
    
            print('\n>>> Proceeding to  SwissModel search...')
            print('------------------------------------\n')
            swiss = no_pdb.copy()
            if len(swiss) > 0:
                print('\n>> Adding SwissModel residue positions...\n')
                swiss.replace({'[]': np.NaN, 'nan-nan': np.NaN, '': np.NaN}, inplace=True)
                swiss = swiss.fillna(np.NaN)
                swiss, no_swiss_models= addSwissModels(swiss, path_to_input_files, path_to_output_files)
                print('\n>> Mapping to SwissModels...\n')
                if len(swiss) > 0:
                    swiss.reset_index(drop=True, inplace=True)
                    swiss = changeUPtoModels(swiss)
                    swiss.reset_index(drop=True, inplace=True)
                    print('\n>> Calculating 3D distances for SwissModels...\n')
                    swiss = isZeroDistance(swiss)
                    swiss = match3DModels(swiss)
                    swiss = selectMaxAnnot(swiss)
                    swiss = swiss.sort_values(by=['datapoint', 'qmean_norm', 'distance', 'hitTotal', 'annotTotal'], ascending=[True, False, True, False, True])
                    swiss = swiss.drop_duplicates(['datapoint'])
                    swiss.replace({'[]': np.NaN, 'hit': 0.0}, inplace=True)
                else:
                    swiss = swiss[SIMPLE_COLS]
    
                if len(no_swiss_models) > 0:
                    no_swiss_models = no_swiss_models[SIMPLE_COLS]
                    no_swiss_models.reset_index(inplace=True, drop=True)
    
            else:
                swiss = swiss[SIMPLE_COLS]
                no_swiss_models = no_pdb.copy()

            if len(no_swiss_models) >0:
                modbase = no_swiss_models.copy()
                print('Proceeding to  Modbase search...')
                print('------------------------------------\n')
                
                modbase = modbase[SIMPLE_COLS]
                modbase.replace({'[]': np.NaN, 'nan-nan': np.NaN, '': np.NaN}, inplace=True)
                modbase = modbase.fillna(np.NaN)
                print('\n>> Adding Modbase residue positions...\n')
                modbase_simple = modbase[['uniprotID', 'wt', 'pos', 'mut','datapoint']]
                modbase_simple = modbase_simple.drop_duplicates(['uniprotID', 'wt', 'pos' ,'mut','datapoint'])
                modbaseOut, no_modbase_models_updated = addModbaseModels(modbase_simple, path_to_input_files, path_to_output_files)

                if len(modbaseOut) > 0:
                    modbase = modbase.merge(modbaseOut, on = ['uniprotID', 'wt', 'pos', 'mut','datapoint'], how = 'left')
                    no_modbase_models_updated['sasa'] = np.NaN
                    modbase.reset_index(inplace=True, drop=True)
                    no_modbase_add = modbase[pd.isna(modbase.coordinates)]
                    modbase = modbase[~pd.isna(modbase.coordinates)]
                    no_modbase_models_updated = pd.concat([no_modbase_models_updated, no_modbase_add])
                    print('\n>> Mapping to Modbase models...\n')
                    modbase = changeUPtoModels(modbase)
                    print('\n>> Calculating 3D distances for Modbase models...\n')
                    modbase = isZeroDistance(modbase)
                    modbase = match3DModels(modbase)
                    modbase = selectMaxAnnot(modbase)
                    modbase = modbase.sort_values(by=['datapoint', 'quality_score', 'distance','hitTotal', 'annotTotal'], ascending=[True, False, True, False, True])
                    modbase = modbase.drop_duplicates(['datapoint'])
                    modbase.replace({'[]': np.NaN, 'hit': 0.0}, inplace=True)

                else:
                    modbase = pd.DataFrame(columns = SIMPLE_COLS)
    
            else:
                no_modbase_models_updated = pd.DataFrame(columns = SIMPLE_COLS)
                modbase= pd.DataFrame(columns = SIMPLE_COLS)
    
            COLS = ['uniprotID', 'wt', 'pos', 'mut', 'datapoint', 'composition', 'polarity', 'volume', 'granthamScore', 'domain', 'domStart', 'domEnd', 'distance',
                    'region', 'crosslink', 'peptide', 'disulfide', 'signalPeptide', 'propeptide', 'naturalVariant', 'nucleotideBinding', 'modifiedResidue', 'site',
                    'caBinding', 'turn', 'transmembrane', 'repeat', 'glycosylation', 'intramembrane', 'metalBinding', 'bindingSite', 'dnaBinding', 'activeSite',
                    'coiledCoil', 'helix', 'mutagenesis', 'zincFinger', 'transitPeptide', 'intMet', 'strand', 'lipidation', 'motif', 'topologicalDomain',
                    'disulfideBinary', 'intMetBinary', 'intramembraneBinary', 'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary', 'nucleotideBindingBinary',
                    'lipidationBinary', 'siteBinary', 'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary', 'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                    'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary', 'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary', 'modifiedResidueBinary', 'zincFingerBinary',
                    'motifBinary', 'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary', 'glycosylationBinary', 'propeptideBinary', 'sasa']
            
            if len(no_modbase_models_updated) == 0:
                no_modbase_models_updated = pd.DataFrame(columns = SIMPLE_COLS)
            no_modbase_models_updated = no_modbase_models_updated[~no_modbase_models_updated.datapoint.isin(modbase.datapoint.to_list())]
            no_modbase_models_updated = no_modbase_models_updated[['uniprotID', 'wt', 'pos', 'mut', 'datapoint']]
            no_modbase_models_updated.pos = no_modbase_models_updated.pos.astype(int)
            no_modbase_models_updated = no_modbase_models_updated.drop_duplicates()

            
            if len(pdb)>0:
                pdb = pdb[COLS]
                pdb['Source'] = 'PDB'
            else:
                pdb = pd.DataFrame()
            if len(swiss) > 0:
                swiss = swiss[COLS]
                swiss['Source'] = 'SWISS-Model'
            else:
                swiss = pd.DataFrame()
            if len(modbase) > 0:
                modbase = modbase[COLS]
                modbase['Source'] = 'Modbase'
            else:
                modbase = pd.DataFrame()
            
    
            # st.write('======PDB==========')
            # st.write(pdb.to_string())
            # st.write('======SWISS==========')
            # st.write(swiss.to_string())
            # st.write('======MODBASE==========')
            # st.write(modbase.to_string())
    

    
            allData = pd.concat([pdb, swiss, modbase])
            allData.reset_index(inplace=True, drop=True)
            allData.replace({np.NaN: ''}, inplace=True)
            # st.write('======ALL DATA==========')
            # st.write(allData.to_string())
            if len(allData)>0:
                allData.distance.replace({-1000: ''}, inplace=True)
    
    
                # Get interface positions from ECLAIR. Download HQ human
                print()
                print('Assigning surface regions...')
                print('------------------------------------\n')
    
                print('Extracting interface residues...\n')
                data_interface = pd.read_csv(path_to_interfaces, sep='\t')
    
                positions = get_interface_positions(data_interface, 'P1', 'P2')
    
                interface_dataframe = pd.DataFrame()
    
                for key, val in positions.items():
                    k = pd.Series((key, str(list(set(val)))))
                    interface_dataframe = interface_dataframe.append(k, ignore_index=True)
                interface_dataframe.columns = ['uniprotID', 'positions']
                final_data = finalTouch(allData)
                final_data = final_data.merge(interface_dataframe, on='uniprotID', how='left')
                final_data.positions = final_data.positions.astype('str')
                for i in final_data.index:
                    if (str(final_data.at[i, 'pos']) in final_data.at[i, 'positions']) and final_data.at[i, 'trsh4'] == 'surface':
                        final_data.at[i, 'threeState_trsh4_HQ'] = 'interface'
                    elif (str(final_data.at[i, 'pos']) not in final_data.at[i, 'positions']) and final_data.at[i, 'trsh4'] == 'surface':
                        final_data.at[i, 'threeState_trsh4_HQ'] = 'surface'
                    elif (str(final_data.at[i, 'pos']) not in final_data.at[i, 'positions']) and final_data.at[i, 'trsh4'] == 'core':
                        final_data.at[i, 'threeState_trsh4_HQ'] = 'core'
                    elif (str(final_data.at[i, 'pos']) in final_data.at[i, 'positions']) and final_data.at[i, 'trsh4'] == 'core':
                        final_data.at[i, 'threeState_trsh4_HQ'] = 'conflict'
                    elif final_data.at[i, 'trsh4'] == 'nan':
                        final_data.at[i, 'threeState_trsh4_HQ'] = 'nan'
    
                final_data.drop(['positions'], axis=1, inplace=True)
    
                fisherResult = pd.read_csv(fisher_path, sep='\t')
                significant_domains = fisherResult.domain.to_list()
                for i in final_data.index:
                    if final_data.at[i, 'domain'] in significant_domains:
                        final_data.at[i, 'domain_fisher'] = final_data.at[i, 'domain']
                    else:
                        final_data.at[i, 'domain_fisher'] = 'NULL'
                print('Final adjustments are being done...\n')
                binaryCols = UNIPROT_ANNOTATION_COLS[-30:]
                final_data = final_data.astype(str)
                final_data.replace({'NaN': 'nan'}, inplace=True)
                for i in final_data.index:
                    for j in binaryCols:
                        final_data[j] = final_data[j].astype('str')
                        if (final_data.at[i, j] == '0') or (final_data.at[i, j] == '0.0'):
                            final_data.at[i, j] = '1'
                        elif final_data.at[i, j] == 'nan':
                            final_data.at[i, j] = '0'
                        elif (final_data.at[i, j] == '1') or (final_data.at[i, j] == '1.0'):
                            final_data.at[i, j] = '2'
    
                annotCols = UNIPROT_ANNOTATION_COLS[:30]
    
                for i in final_data.index:
                    for annot in annotCols:
                        binaryName = str(annot) + 'Binary'
                        if final_data.at[i, binaryName] == '2':
                            final_data.at[i, annot] = '0.0'
                final_data.rename(
                    columns={'uniprotID': 'prot_uniprotAcc', 'wt': 'wt_residue', 'pos': 'position', 'mut': 'mut_residue',
                             'datapoint': 'meta_merged', 'datapoint_disease': 'meta-lab_merged', 'label': 'source_db',
                             'family': 'prot_family', 'domain': 'domains_all', 'domain_fisher': 'domains_sig',
                             'distance': 'domains_3Ddist', 'threeState_trsh4_HQ': 'location_3state',
                             'disulfideBinary': 'disulfide_bin', 'intMetBinary': 'intMet_bin',
                             'intramembraneBinary': 'intramembrane_bin',
                             'naturalVariantBinary': 'naturalVariant_bin', 'dnaBindingBinary': 'dnaBinding_bin',
                             'activeSiteBinary': 'activeSite_bin',
                             'nucleotideBindingBinary': 'nucleotideBinding_bin', 'lipidationBinary': 'lipidation_bin',
                             'siteBinary': 'site_bin',
                             'transmembraneBinary': 'transmembrane_bin', 'crosslinkBinary': 'crosslink_bin',
                             'mutagenesisBinary': 'mutagenesis_bin',
                             'strandBinary': 'strand_bin', 'helixBinary': 'helix_bin', 'turnBinary': 'turn_bin',
                             'metalBindingBinary': 'metalBinding_bin',
                             'repeatBinary': 'repeat_bin', 'topologicalDomainBinary': 'topologicalDomain_bin',
                             'caBindingBinary': 'caBinding_bin',
                             'bindingSiteBinary': 'bindingSite_bin', 'regionBinary': 'region_bin',
                             'signalPeptideBinary': 'signalPeptide_bin',
                             'modifiedResidueBinary': 'modifiedResidue_bin', 'zincFingerBinary': 'zincFinger_bin',
                             'motifBinary': 'motif_bin',
                             'coiledCoilBinary': 'coiledCoil_bin', 'peptideBinary': 'peptide_bin',
                             'transitPeptideBinary': 'transitPeptide_bin',
                             'glycosylationBinary': 'glycosylation_bin', 'propeptideBinary': 'propeptide_bin',
                             'disulfide': 'disulfide_dist', 'intMet': 'intMet_dist',
                             'intramembrane': 'intramembrane_dist', 'naturalVariant': 'naturalVariant_dist',
                             'dnaBinding': 'dnaBinding_dist', 'activeSite': 'activeSite_dist',
                             'nucleotideBinding': 'nucleotideBinding_dist', 'lipidation': 'lipidation_dist',
                             'site': 'site_dist',
                             'transmembrane': 'transmembrane_dist', 'crosslink': 'crosslink_dist',
                             'mutagenesis': 'mutagenesis_dist', 'strand': 'strand_dist', 'helix': 'helix_dist',
                             'turn': 'turn_dist',
                             'metalBinding': 'metalBinding_dist', 'repeat': 'repeat_dist',
                             'topologicalDomain': 'topologicalDomain_dist', 'caBinding': 'caBinding_dist',
                             'bindingSite': 'bindingSite_dist', 'region': 'region_dist',
                             'signalPeptide': 'signalPeptide_dist', 'modifiedResidue': 'modifiedResidue_dist',
                             'zincFinger': 'zincFinger_dist', 'motif': 'motif_dist', 'coiledCoil': 'coiledCoil_dist',
                             'peptide': 'peptide_dist', 'transitPeptide': 'transitPeptide_dist',
                             'glycosylation': 'glycosylation_dist', 'propeptide': 'propeptide_dist'}, inplace=True)
    
                final_data = final_data[
                    ['prot_uniprotAcc', 'wt_residue', 'mut_residue', 'position','Source', 'meta_merged', 'composition', 'polarity',
                     'volume',
                     'granthamScore', 'domains_all',
                     'domains_sig', 'domains_3Ddist', 'sasa', 'location_3state', 'disulfide_bin', 'intMet_bin',
                     'intramembrane_bin', 'naturalVariant_bin', 'dnaBinding_bin',
                     'activeSite_bin', 'nucleotideBinding_bin', 'lipidation_bin', 'site_bin',
                     'transmembrane_bin', 'crosslink_bin', 'mutagenesis_bin', 'strand_bin',
                     'helix_bin', 'turn_bin', 'metalBinding_bin', 'repeat_bin',
                     'caBinding_bin', 'topologicalDomain_bin', 'bindingSite_bin',
                     'region_bin', 'signalPeptide_bin', 'modifiedResidue_bin',
                     'zincFinger_bin', 'motif_bin', 'coiledCoil_bin', 'peptide_bin',
                     'transitPeptide_bin', 'glycosylation_bin', 'propeptide_bin', 'disulfide_dist', 'intMet_dist',
                     'intramembrane_dist',
                     'naturalVariant_dist', 'dnaBinding_dist', 'activeSite_dist',
                     'nucleotideBinding_dist', 'lipidation_dist', 'site_dist',
                     'transmembrane_dist', 'crosslink_dist', 'mutagenesis_dist',
                     'strand_dist', 'helix_dist', 'turn_dist', 'metalBinding_dist',
                     'repeat_dist', 'caBinding_dist', 'topologicalDomain_dist',
                     'bindingSite_dist', 'region_dist', 'signalPeptide_dist',
                     'modifiedResidue_dist', 'zincFinger_dist', 'motif_dist',
                     'coiledCoil_dist', 'peptide_dist', 'transitPeptide_dist',
                     'glycosylation_dist', 'propeptide_dist']]
                # Imputation
                if (impute == 'True') or (impute == 'true') or (impute == True):
                    filler = [17.84, 30.8, 24.96, 13.12, 23.62, 18.97, 20.87, 29.59, 20.7, 12.7, 22.85, 17.21, 9.8, 9, 15.99,
                              16.82,
                              20.46, 24.58, 9.99, 17.43, 20.08, 30.91, 20.86, 22.14, 21.91, 28.45, 17.81, 25.12, 20.33, 22.36]
                    col_index = 0
                    for col_ in final_data.columns[-30:]:
                        final_data[col_] = final_data[col_].fillna(filler[col_index])
                        final_data[col_] = final_data[col_].replace({'nan': filler[col_index]})
                        col_index += 1
                    final_data['domains_3Ddist'] = final_data['domains_3Ddist'].fillna(24.5)
                    final_data['sasa'] = final_data['sasa'].fillna(29.5)
                    final_data['location_3state'] = final_data['location_3state'].fillna('unknown')
                elif (impute == 'False') or (impute == 'false'):
                    pass
                final_data = final_data.replace({'nan': np.NaN})
                final_data.domains_all = final_data.domains_all.replace({-1: 'NULL'})
    
                # ready.to_csv(path_to_output_files / 'featurevector_pdb.txt', sep='\t', index=False)
                if len(final_data) == 0:
                    print(
                        'No feature vector could be produced for input data. Please check the presence of a structure for the input proteins.')
                final_data.to_csv(path_to_output_files / 'featurevector_pdb.txt', sep='\t', index=False)
        
                print('Feature vector successfully created...')
                end = timer()
                hours, rem = divmod(end - start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                if len(no_modbase_models_updated) >0 and (len(no_modbase_models_updated) !=org_len):
                    st.write(f'{len(no_modbase_models_updated)} of {org_len} datapoins could not be mapped to any structures.')
                    st.write(f'{org_len-len(no_modbase_models_updated)} of {org_len} datapoins were mapped to a structure.')
                elif len(no_modbase_models_updated) == org_len:
                    st.write(f'0 of {org_len} datapoins could not be mapped to any structures. Feature vector could not be created.')

                return final_data
            elif len(no_modbase_models_updated) >0 and (len(no_modbase_models_updated) !=org_len):
                st.write(f'{len(no_modbase_models_updated)} of {org_len} datapoins could not be mapped to any structures.')
                st.write(f'{org_len-len(no_modbase_models_updated)} of {org_len} datapoins were mapped to a structure.')
            elif len(no_modbase_models_updated) == org_len:
                st.write(f'0 of {org_len} datapoins could not be mapped to any structures. Feature vector could not be created.')
