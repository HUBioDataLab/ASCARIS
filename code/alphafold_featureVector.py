# IMPORT NECESSARY MODULES AND LIBRARIES
from timeit import default_timer as timer
import xml.etree.ElementTree as ET
from collections import Counter
from bs4 import BeautifulSoup
from io import StringIO
from decimal import *
import pandas as pd
import requests as r
import os.path as op
from pathlib import Path
import subprocess
import argparse
import ssbio.utils
import warnings
import sys
import pathlib
import os, glob
import math
import ssbio
import ssl
import gzip
import ast
import itertools

from Bio.Align import substitution_matrices
from Bio.PDB.Polypeptide import *
from Bio.PDB import PDBList
from Bio import Align
from Bio import SeqIO
from Bio.PDB import *
import numpy as np




# FUNCTIONS
from calc_pc_property import *
from add_domains import *
from add_annotations import *
from add_structure import *
from add_alignment import *
from manage_files import *
from add_3Dalignment import *
from add_sasa import *
from standard import *
from add_interface_pos import *
from standard import *
from uniprotSequenceMatch import uniprotSequenceMatch
from process_input import clean_data
from alphafold_model import *


def alphafold(input_set, mode, impute):
    start = timer()
    # Necessary lists
    annotation_list = ['disulfide', 'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                       'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink', 'mutagenesis', 'strand',
                       'helix', 'turn', 'metalBinding', 'repeat', 'topologicalDomain', 'caBinding', 'bindingSite',
                       'region',
                       'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil', 'peptide',
                       'transitPeptide', 'glycosylation', 'propeptide']

    change_names = {'Disulfide bond': 'disulfide', 'Initiator methionine': 'intMet',
                    'Natural variant': 'naturalVariant',
                    'DNA binding': 'dnaBinding',
                    'Active site': 'activeSite', 'Nucleotide binding': 'nucleotideBinding', 'Lipidation': 'lipidation',
                    'Site': 'site', 'Transmembrane': 'transmembrane', 'Cross-link': 'crosslink',
                    'Mutagenesis': 'mutagenesis', 'Beta strand': 'strand', 'Helix': 'helix', 'Turn': 'turn',
                    'Metal binding': 'metalBinding', 'Repeat': 'repeat',
                    'Topological domain': 'topologicalDomain', 'Calcium binding': 'caBinding',
                    'Binding site': 'bindingSite',
                    'Region': 'region', 'Signal peptide': 'signalPeptide', 'Modified residue': 'modifiedResidue',
                    'Zinc finger': 'zincFinger', 'Motif': 'motif', 'Coiled coil': 'coiledCoil', 'Peptide': 'peptide',
                    'Transit peptide': 'transitPeptide', 'Glycosylation': 'glycosylation', 'Propeptide': 'propeptide',
                    'Intramembrane': 'intramembrane'}


    ## Standardizing input
    data = clean_data(input_set)

    path_to_input_files, path_to_output_files, path_to_domains, fisher_path, path_to_interfaces, alphafold_path, alphafold_summary= manage_files(mode)
    out_path = path_to_output_files / 'log.txt'
    sys.stdout = open(out_path, 'w')
    print('Creating directories...')
    file_base = str(Path(alphafold_path / '*'))
    file_str = glob.glob(file_base)[0].split('-')[-1].split('.')[0]
    ## Physicochemical properties
    print('Adding physicochemical properties...\n')
    data = add_physicochemical(data)

    ## Domains
    print('Adding domains\n')
    data = add_domains(data, path_to_domains)

    ## Processing data frame
    data = data.astype(str)
    data = data.replace({'NaN': np.NaN, 'nan': np.NaN})
    data.domain = data.domain.replace({np.NaN: '-1'})  # Fill -1 if NaN - standardization.
    data.domStart = data.domStart.replace({np.NaN: '-1'})
    data.domEnd = data.domEnd.replace({np.NaN: '-1'})
    data.distance = data.distance.replace({np.NaN: '-1'})
    fisherResult = pd.read_csv(fisher_path, sep='\t')
    significant_domains = fisherResult.domain.to_list()

    data = data.reset_index()
    data = data.drop(columns=['index'])

    ## not_match_in_uniprot : Data points not matched to UniProt sequence
    ## uniprot_matched: Data points matched to UniProt sequence. Proceed with this data frame
    ## canonical_fasta : Dataframe including canonical sequence for the protein of interest. Obtained from UniProt.
    ## isoform_fasta: Dataframe including isoform sequences for the protein of interest. Obtained from UniProt.
    not_match_in_uniprot, uniprot_matched, canonical_fasta, isoform_fasta = uniprotSequenceMatch(data)

    not_match_in_uniprot = not_match_in_uniprot.reset_index().drop(['index'], axis=1)

    for key in change_names.keys():
        not_match_in_uniprot[key] = ''
    not_match_in_uniprot = not_match_in_uniprot.rename(columns=change_names)
    uniprot_matched = add_annotations(uniprot_matched)

    for w in uniprot_matched.index:
        for q in annotation_list:
            per_protein = []
            if uniprot_matched.at[w, q] != 'nan':
                fix = ast.literal_eval(uniprot_matched.at[w, q])
                for z in fix:
                    if '-' in z:
                        per_protein += np.arange(int(z.split('-')[0]), int(z.split('-')[1])+1,1).tolist()
                    else:
                        try:
                            per_protein.append(int(z))
                        except:
                            ValueError
                uniprot_matched.at[w, q] = per_protein
            else:
                uniprot_matched.at[w, q] = 'nan'
    uniprot_matched = uniprot_matched.rename(columns=change_names)
    uniprot_matched['wt_sequence_match'] = uniprot_matched['wt_sequence_match'].astype(str)


    ## Avoiding downloading files for SASA calculation if already downloaded.

    existing_free_sasa = list(Path(path_to_output_files / 'freesasa_files').glob("*"))
    existing_free_sasa = [str(i) for i in existing_free_sasa]
    existing_free_sasa = [i.split('/')[-1].split('.')[0] for i in existing_free_sasa]
    ## Decide if the wild type amino acid is on canonical or isoform sequence. Selected sequence will be used for the
    ## sequence alignment.
    for i in uniprot_matched.index:
        if len(uniprot_matched.at[i, 'uniprotSequence']) >= int(uniprot_matched.at[i, 'pos']):
            wt = uniprot_matched.at[i, 'wt']
            can = str(uniprot_matched.at[i, 'uniprotSequence'])[int(uniprot_matched.at[i, 'pos']) - 1]
            if wt == can:
                uniprot_matched.at[i, 'wt_sequence_match'] = 'm'
            elif wt != can:
                isoList = isoform_fasta[
                    isoform_fasta['uniprotID'] == uniprot_matched.at[i, 'uniprotID']].isoformSequence.to_list()
                for k in isoList:
                    if len(k) >= int(uniprot_matched.at[i, 'pos']):
                        resInIso = k[int(int(uniprot_matched.at[i, 'pos']) - 1)]
                        if wt == resInIso:
                            whichIsoform = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                            uniprot_matched.at[i, 'wt_sequence_match'] = 'i'
                            uniprot_matched.at[i, 'whichIsoform'] = whichIsoform
                            break

        elif len(uniprot_matched.at[i, 'uniprotSequence']) < int(uniprot_matched.at[i, 'pos']):
            isoList = isoform_fasta[
                isoform_fasta['uniprotID'] == uniprot_matched.at[i, 'uniprotID']].isoformSequence.to_list()
            for k in isoList:
                if len(k) >= int(uniprot_matched.at[i, 'pos']):
                    resInIso = k[int(int(uniprot_matched.at[i, 'pos']) - 1)]
                    wt = uniprot_matched.at[i, 'wt']
                    if wt == resInIso:
                        whichIsoform = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                        uniprot_matched.at[i, 'wt_sequence_match'] = 'i'
                        uniprot_matched.at[i, 'whichIsoform'] = whichIsoform
                        break



    uniprot_matched = uniprot_matched.replace({'nan': np.NaN})
    for annot in ['Domain', 'Alternative sequence', 'Chain', 'Sequence conflict', 'Compositional bias']:
        try:
            uniprot_matched = uniprot_matched.drop(columns=annot)
        except:
            KeyError

    print('You have %d data points that failed to match a UniProt Sequence\nProceeding with %d remaining...\n'
          % (len(not_match_in_uniprot.drop_duplicates(['datapoint'])),
             len(uniprot_matched.drop_duplicates(['datapoint']))))

    ## Adding interface residue information.

    data_interface = pd.read_csv(path_to_interfaces, sep='\t')
    interface_positions = get_interface_positions(data_interface, 'P1', 'P2')

    interface_dataframe = pd.DataFrame()
    for key, val in interface_positions.items():
        k = pd.Series((key, str(list(set(val)))))
        interface_dataframe = interface_dataframe.append(k, ignore_index=True)
    interface_dataframe.columns = ['uniprotID', 'interface_positions']

    uniprot_matched = uniprot_matched.merge(interface_dataframe, on='uniprotID', how='left')
    uniprot_matched.interface_positions = uniprot_matched.interface_positions.astype('str')

    ## PDB info file is pre-generated for time concerns. Includes summary data of AlphaFold structures.
    ## With new updates, can be updated separately.

    pdb_info = pd.read_csv(alphafold_summary, sep='\t')

    ## Keeping how many models each AlphaFold structure has.
    model_count = modelCount(alphafold_path)
    for k, v in model_count.items():
        model_count[k] = int(v / 2)  # two types of files for each file.
    uniprot_matched = uniprot_matched.astype(str)
    uniprot_matched.domStart = uniprot_matched.domStart.astype(float)
    uniprot_matched.domEnd = uniprot_matched.domEnd.astype(float)
    uniprot_matched.domStart = uniprot_matched.domStart.astype(int)
    uniprot_matched.domEnd = uniprot_matched.domEnd.astype(int)



    ## Main part to add annotation information, align sequences, finding distances

    for i in uniprot_matched.index:
        print('Processing', i, 'of', len(uniprot_matched))
        if len(uniprot_matched.at[i, 'uniprotSequence']) >= int(uniprot_matched.at[i, 'pos']):
            wt = uniprot_matched.at[i, 'wt']
            can = str(uniprot_matched.at[i, 'uniprotSequence'])[int(uniprot_matched.at[i, 'pos']) - 1]
            ## Information about whether the mutation is found on the canonical or isoform sequence.

            if wt == can:
                uniprot_matched.at[i, 'wt_sequence_match'] = 'm'
            elif wt != can:
                isoList = isoform_fasta[
                    isoform_fasta['uniprotID'] == uniprot_matched.at[i, 'uniprotID']].isoformSequence.to_list()
                for k in isoList:
                    if len(k) >= int(uniprot_matched.at[i, 'pos']):
                        resInIso = k[int(int(uniprot_matched.at[i, 'pos']) - 1)]
                        if wt == resInIso:
                            whichIsoform = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                            uniprot_matched.at[i, 'wt_sequence_match'] = 'i'
                            uniprot_matched.at[i, 'whichIsoform'] = whichIsoform
                            break
        elif len(uniprot_matched.at[i, 'uniprotSequence']) < int(uniprot_matched.at[i, 'pos']):
            isoList = isoform_fasta[
                isoform_fasta['uniprotID'] == uniprot_matched.at[i, 'uniprotID']].isoformSequence.to_list()
            for k in isoList:
                if len(k) >= int(uniprot_matched.at[i, 'pos']):
                    resInIso = k[int(int(uniprot_matched.at[i, 'pos']) - 1)]
                    wt = uniprot_matched.at[i, 'wt']
                    if wt == resInIso:
                        whichIsoform = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                        uniprot_matched.at[i, 'wt_sequence_match'] = 'i'
                        uniprot_matched.at[i, 'whichIsoform'] = whichIsoform
                        break
        uniprotID = uniprot_matched.at[i, 'uniprotID']
        datapoint = uniprot_matched.at[i, 'datapoint']

        for k in annotation_list:
            txt = k + 'Binary'

            if (str(uniprot_matched.at[i, txt]) == '0') or (str(uniprot_matched.at[i, txt]) == '0.0'):
                uniprot_matched.at[i, txt] = '1'
            elif (str(uniprot_matched.at[i, txt]).lower() == 'nan') | (str(uniprot_matched.at[i, txt]) == np.NaN) :
                uniprot_matched.at[i, txt] = '0'
            elif (str(uniprot_matched.at[i, txt]) == '1') or (str(uniprot_matched.at[i, txt]) == '1.0'):
                uniprot_matched.at[i, txt] = '2'
        ## Search in all models.
        models_for_protein = [val for key, val in model_count.items() if
                              uniprotID in key.split(';')]  # We have this many models for the protein.
        which_model_mutation = which_model(
            int(uniprot_matched.at[i, 'pos']))  # List of models in which the mutation can be found.
        models_for_all_annotations = {}
        for annot in annotation_list:
            if len(uniprot_matched.at[i, annot]) != 0 and type(uniprot_matched.at[i, annot]) != list:
                uniprot_matched.at[i, annot] = list(
                    map(str.strip, uniprot_matched.at[i, annot].strip('][').replace('"', '').split(',')))
            models_for_annotations = {}  # Recording which position is found in which model file.
            for annot_position in uniprot_matched.at[i, annot]:
                if annot_position != 'nan' and annot_position != '':
                    models_for_that_position = which_model(int(annot_position))
                else:
                    models_for_that_position = {}
                for key, val in models_for_that_position.items():
                    if key not in models_for_annotations.keys():
                        models_for_annotations[key] = [val]
                    else:
                        models_for_annotations[key] += [val]
            models_for_all_annotations[annot] = models_for_annotations
        new_dict = {}
        for key, val in models_for_all_annotations.items():
            subdict = {k: v for k, v in val.items() if k in which_model_mutation}
            subdict = dict(sorted(subdict.items()))
            new_dict[key] = subdict
        new_dict = reduce_model_dict(new_dict)
        models_we_need = list(set(itertools.chain.from_iterable(
            [list(ov.keys()) for ok, ov in new_dict.items()])))  # Read models with these numbers
        info_per_model = {}  # her bir datapoint için baştan yazılıyor.
        dist_of_annots = {}
        all_domain_distances = []

        for mod in models_we_need:
            print('---------PRINTING FOR MODEL--------', mod)
            dist_of_annots[str(mod)] = {}
            info_per_model[mod] = {}
            info_per_model[mod]['datapoint'] = datapoint
            identifier = uniprot_matched.at[i, 'uniprotSequence']
            try:
                pdbSequence = pdb_info.loc[(pdb_info.uniprotID == uniprotID) & (
                        pdb_info.model_num == mod)].sequence.item()
            except:
                ValueError
                pdbSequence = 'nan'
            if pdbSequence != 'nan':  # The number in models we need might not be present for that protein. Preventng error.
                pdbSequence = pdb_info.loc[(pdb_info.uniprotID == uniprotID) & (pdb_info.model_num == mod)].sequence.item()
                alignment_list = do_alignment(uniprot_matched.at[i, 'datapoint'], uniprot_matched.at[i, 'uniprotSequence'],
                                              pdbSequence, Path(path_to_output_files / 'alignment_files'))
                pdb_alignStatus = mutation_position_on_pdb(alignment_list, uniprot_matched.at[i, 'pos'])[0]
                info_per_model[mod]['pdb_alignStatus'] = pdb_alignStatus
                mutationPositionOnPDB = mutation_position_on_pdb(alignment_list, uniprot_matched.at[i, 'pos'])[1]
                info_per_model[mod]['mutationPositionOnPDB'] = mutationPositionOnPDB
                startGap = mutation_position_on_pdb(alignment_list, uniprot_matched.at[i, 'pos'])[2]
                info_per_model[mod]['startGap'] = startGap
                alignment_to_use = mutation_position_on_pdb(alignment_list, uniprot_matched.at[i, 'pos'])[3]
                for annot in annotation_list:
                    if new_dict[annot] == {}:
                        annotation_pos_on_pdb_ = []
                    else:
                        try:
                            annotation_pos_on_pdb_ = annotation_pos_on_pdb(new_dict[annot][mod], startGap, alignment_to_use,
                                                                           identifier)
                        except:
                            KeyError
                    info_per_model[mod][annot] = annotation_pos_on_pdb_

                pdb_path = Path(f'{alphafold_path}/AF-{uniprotID}-F{mod}-{file_str}.pdb.gz')

                if get_alignments_3D(uniprotID, mod, pdb_path, pdbSequence, 'nan', 'nan', 'nan', mode, Path(path_to_output_files / '3D_alignment'),
                                     'gzip') != None:
                    alignments, coords, resnums_for_sasa = get_alignments_3D(uniprotID, mod, pdb_path, pdbSequence, 'nan',
                                                                            'nan', 'nan', mode, Path(path_to_output_files / '3D_alignment'),
                                                                            'gzip')
                    alignments = alignments[0]

                    calculate_freesasa(uniprotID, mod, existing_free_sasa, alphafold_path, path_to_output_files)
                    if (mutationPositionOnPDB != 'nan'):
                        if (int(mutationPositionOnPDB) <= 1400):
                            try:
                                coordMut = get_coords(mutationPositionOnPDB, alignments, coords, resnums_for_sasa, mode)[0]
                            except:
                                ValueError
                                coordMut = 'nan'
                        else:
                            coordMut = np.NaN

                        sasa_pos = get_coords(mutationPositionOnPDB, alignments, coords, resnums_for_sasa, mode)[2]
                        sasa_val = sasa('alphafold', 'nan', uniprotID, sasa_pos, uniprot_matched.at[i, 'wt'], mode,
                                        path_to_output_files, file_type='gzip')

                        if sasa_val != None:
                            uniprot_matched.at[i, 'sasa'] = sasa_val
                    else:
                        coordMut = 'nan'
                        sasa_val = 'nan'
                        uniprot_matched.at[i, 'sasa'] = sasa_val

                    domainPositionOnPDB_list = list(
                        range(int(uniprot_matched.at[i, 'domStart']), int(uniprot_matched.at[i, 'domEnd'])))
                    domain_distances = []
                    if len(domainPositionOnPDB_list) != 0:
                        for domain_ in domainPositionOnPDB_list:
                            coordDomain = get_coords(domain_, alignments, coords, resnums_for_sasa, mode)[0]
                            distance_dom = float(find_distance(coordMut,
                                                               coordDomain))  # bu bir anotasyonun bir modeldeki bir tane pozisyonu için.
                            domain_distances.append(distance_dom)
                        minimum_domain = min(domain_distances)  # minimum for one model.
                    else:
                        minimum_domain = np.NaN
                    all_domain_distances.append(minimum_domain)
                    list_dist_of_annots = []
                    for key, val in info_per_model.items():
                        modNum = key
                        min_annots = {}  # Write from scratch for each annotation.

                        if modNum == mod:
                            for label, annotPos in val.items():  # For each annotation type, calculate all distances of the annot positions.
                                if label in annotation_list:
                                    all_annot_distance_per_model = []  # All distances of an annoation in hat model
                                    for annot_position in annotPos:
                                        if (annot_position != 'nan'):
                                            if (int(annot_position) <= 1400):
                                                coordAnnot = \
                                                    get_coords(annot_position, alignments, coords, resnums_for_sasa, mode)[
                                                        0]
                                                distance = float(find_distance(coordMut,
                                                                               coordAnnot))  # bu bir anotasyonun bir modeldeki bir tane pozisyonu için.
                                                all_annot_distance_per_model.append(distance)
                                    if all_annot_distance_per_model != []:
                                        all_annot_distance_per_model = [float(i) for i in all_annot_distance_per_model]
                                        try:
                                            minimum_position = float(min(all_annot_distance_per_model))
                                        except:
                                            ValueError
                                            minimum_position = 'nan'
                                        min_annots[label] = float(
                                            minimum_position)  # Minimum of the annotation in this model.
                        if min_annots != {}:
                            list_dist_of_annots.append(min_annots)
                    dist_of_annots[str(
                        mod)] = list_dist_of_annots  # Getting minimum of all possible models
                #                uniprot_matched.at[i, annotation_type] = minimum_position
                else:

                    print('Model File Not Found')
                    uniprot_matched.at[i, 'sasa'] = np.NaN



        if len(all_domain_distances) != 0:
            uniprot_matched.at[i, 'domaindistance3D'] = min(all_domain_distances)
        else:
            uniprot_matched.at[i, 'domaindistance3D'] = np.NaN
        dist_of_annots_min_of_all = {}
        flat = [item for sublist in list(dist_of_annots.values()) for item in sublist]
        for f in flat:
            for key, val in f.items():
                if key not in dist_of_annots_min_of_all.keys():
                    dist_of_annots_min_of_all[key] = val
                elif (key in dist_of_annots_min_of_all.keys()) & (float(dist_of_annots_min_of_all[key]) > float(val)):
                    dist_of_annots_min_of_all[key] = val
        key_list = []
        for key, val in dist_of_annots_min_of_all.items():
            uniprot_matched.at[i, key] = val
            key_list.append(key)
        remaining = list(set(annotation_list) - set(key_list))

        for rem in remaining:
            uniprot_matched.at[i, rem] = ''
        uniprot_matched.at[i, 'distances'] = [dist_of_annots]

        if (uniprot_matched.at[i, 'sasa'] != None) & (uniprot_matched.at[i, 'sasa'] != np.NaN) & (
                str(uniprot_matched.at[i, 'sasa']) != 'nan'):
            if '*' in uniprot_matched.at[i, 'sasa']:
                uniprot_matched.at[i, 'sasa'] = uniprot_matched.at[i, 'sasa'].split('*')[0]
        try:
            uniprot_matched.at[i, 'sasa'] = float(uniprot_matched.at[i, 'sasa'].strip())
        except:
            TypeError

        if float(uniprot_matched.at[i, 'sasa']) < 5:
            uniprot_matched.at[i, 'trsh4'] = 'core'
        elif float(uniprot_matched.at[i, 'sasa']) >= 5:
            uniprot_matched.at[i, 'trsh4'] = 'surface'
        elif str(uniprot_matched.at[i, 'sasa']) == 'nan':
            uniprot_matched.at[i, 'trsh4'] = 'nan'
        else:
            uniprot_matched.at[i, 'trsh4'] = 'nan'
        if (str(uniprot_matched.at[i, 'pos']) in uniprot_matched.at[i, 'interface_positions']) and uniprot_matched.at[
            i, 'trsh4'] == 'surface':
            uniprot_matched.at[i, 'threeState_trsh4_HQ'] = 'interface'
        elif (str(uniprot_matched.at[i, 'pos']) not in uniprot_matched.at[i, 'interface_positions']) and uniprot_matched.at[
            i, 'trsh4'] == 'surface':
            uniprot_matched.at[i, 'threeState_trsh4_HQ'] = 'surface'
        elif (str(uniprot_matched.at[i, 'pos']) not in uniprot_matched.at[i, 'interface_positions']) and uniprot_matched.at[
            i, 'trsh4'] == 'core':
            uniprot_matched.at[i, 'threeState_trsh4_HQ'] = 'core'
        elif (str(uniprot_matched.at[i, 'pos']) in uniprot_matched.at[i, 'interface_positions']) and uniprot_matched.at[
            i, 'trsh4'] == 'core':
            uniprot_matched.at[i, 'threeState_trsh4_HQ'] = 'conflict'
        elif uniprot_matched.at[i, 'trsh4'] == 'nan':
            uniprot_matched.at[i, 'threeState_trsh4_HQ'] = 'nan'
        if uniprot_matched.at[i, 'domain'] in significant_domains:
            uniprot_matched.at[i, 'domain_fisher'] = uniprot_matched.at[i, 'domain']
        else:
            uniprot_matched.at[i, 'domain_fisher'] = 'NULL'
        uniprot_matched = uniprot_matched.round(2)
        uniprot_matched = uniprot_matched.astype(str)

    uniprot_matched[ 'domain'] = uniprot_matched['domain'].replace({'-1': 'NULL'})
    uniprot_matched = uniprot_matched.drop_duplicates()
    uniprot_matched.rename(
        columns={'uniprotID': 'prot_uniprotAcc', 'wt': 'wt_residue', 'pos': 'position', 'mut': 'mut_residue',
                 'datapoint': 'meta_merged', 'datapoint_disease': 'meta-lab_merged', 'label': 'source_db',
                 'family': 'prot_family', 'domain': 'domains_all', 'domain_fisher': 'domains_sig',
                 'domaindistance3D': 'domains_3Ddist', 'threeState_trsh4_HQ': 'location_3state',
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
                 'nucleotideBinding': 'nucleotideBinding_dist', 'lipidation': 'lipidation_dist', 'site': 'site_dist',
                 'transmembrane': 'transmembrane_dist', 'crosslink': 'crosslink_dist',
                 'mutagenesis': 'mutagenesis_dist', 'strand': 'strand_dist', 'helix': 'helix_dist', 'turn': 'turn_dist',
                 'metalBinding': 'metalBinding_dist', 'repeat': 'repeat_dist',
                 'topologicalDomain': 'topologicalDomain_dist', 'caBinding': 'caBinding_dist',
                 'bindingSite': 'bindingSite_dist', 'region': 'region_dist',
                 'signalPeptide': 'signalPeptide_dist', 'modifiedResidue': 'modifiedResidue_dist',
                 'zincFinger': 'zincFinger_dist', 'motif': 'motif_dist', 'coiledCoil': 'coiledCoil_dist',
                 'peptide': 'peptide_dist', 'transitPeptide': 'transitPeptide_dist',
                 'glycosylation': 'glycosylation_dist', 'propeptide': 'propeptide_dist'}, inplace=True)

    uniprot_matched = uniprot_matched[
        ['prot_uniprotAcc', 'wt_residue', 'mut_residue', 'position', 'meta_merged', 'composition', 'polarity', 'volume',
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
    uniprot_matched = uniprot_matched.reset_index()
    uniprot_matched = uniprot_matched.drop(columns = {'index'})
    # Imputation
    if (impute == 'True') or (impute == 'true'):
        filler = [20.71, 46.67, 28.13,15.5, 35.94, 21.84, 25.15, 45.15, 29.81, 29.91, 34.67, 24.72, 10.66,11.55, 13.02,
                  21.54,27.42, 38.39, 30.44, 20.9, 25.82, 46.12, 32.1, 35.96, 35.86, 37.88, 19.09, 35.2, 26.95, 37.48]
        col_index = 0

        for col_ in uniprot_matched.columns[-30:]:
            uniprot_matched[col_] = uniprot_matched[col_].fillna(filler[col_index])
            uniprot_matched[col_] = uniprot_matched[col_].replace({'nan': filler[col_index]})
            uniprot_matched[col_] = uniprot_matched[col_].replace({'': filler[col_index]})
            """
            if uniprot_matched[col_].values == '':
                uniprot_matched[col_] = filler[col_index]
            """
            col_index += 1

        uniprot_matched['domains_3Ddist'] = uniprot_matched['domains_3Ddist'].fillna(29.78)
        uniprot_matched['sasa'] = uniprot_matched['sasa'].fillna(35.6)
        uniprot_matched['location_3state'] = uniprot_matched['location_3state'].fillna('unknown')
    elif (impute == 'False') or (impute == 'false'):
        pass
    uniprot_matched = uniprot_matched.replace({'nan': np.NaN})
    uniprot_matched = uniprot_matched.replace({'['']': np.NaN})
    uniprot_matched.to_csv(path_to_output_files / 'featurevector_alphafold.txt', index=False, sep='\t')
    if len(uniprot_matched) == 0:
        print(
            'No feature vector could be produced for input data. Please check the presence of a structure for the input proteins.')

    print('Feature vector successfully created...')
    end = timer()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    sys.stdout.close()
    return uniprot_matched
