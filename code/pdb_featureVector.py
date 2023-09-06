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
from Bio.Align import substitution_matrices
from Bio.PDB.Polypeptide import *
from Bio.PDB import PDBList
from Bio import Align
from Bio import SeqIO
from Bio.PDB import *
warnings.filterwarnings("ignore")
start = timer()

# FUNCTIONS



# FUNCTIONS
from calc_pc_property import *
from add_domains import *
from add_annotations import *
from add_sequence import *
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



def pdb(input_set, mode, impute):
    aligner = Align.PairwiseAligner()
    """
    STEP 1
    Get input data as a console input.
    Add datapoint identifier and remove non-standard input.
    """
    data = clean_data(input_set)
    path_to_input_files, path_to_output_files, path_to_domains, fisher_path, path_to_interfaces, buffer =  manage_files(mode)
    out_path = path_to_output_files / 'log.txt'
    sys.stdout = open(out_path, 'w')
    print('Creating directories...')

    annotation_list = ['disulfide', 'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                       'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink', 'mutagenesis', 'strand',
                       'helix', 'turn', 'metalBinding', 'repeat', 'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                       'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil', 'peptide',
                       'transitPeptide', 'glycosylation', 'propeptide']

    print('Feature vector generation started...\n')
    if len(data) == 0:
        print('Feature vectore generation terminated.')
    else:
        """
        STEP 2
        Add physicochemical properties.
        """
        print('Adding physicochemical properties...\n')

        data = add_physicochemical(data)

        """
        STEP 3
        Add domain-related information.
        """
        print('Adding domains\n')

        data = add_domains(data, path_to_domains)

        data = data.astype(str)
        data = data.replace({'NaN': 'nan'})
        data.domain = data.domain.replace({'nan': '-1'})
        data.domStart = data.domStart.replace({'nan': '-1'})
        data.domEnd = data.domEnd.replace({'nan': '-1'})
        data.distance = data.distance.replace({'nan': '-1'})

        """
        STEP 4
        Retrieve canonical and isoform UniProt sequences.
        Add to the data frame.
        """
        print('Retrieving UniProt sequences...\n')

        canonical_fasta = pd.DataFrame(columns=['uniprotID', 'uniprotSequence'])
        up_list = list(set(data['uniprotID'].to_list()))
        for i in range(len(up_list)):
            canonical_fasta.at[i, 'uniprotSequence'] = get_uniprot_seq(up_list[i])
            canonical_fasta.at[i, 'uniprotID'] = up_list[i]

        canonical_fasta = canonical_fasta.drop_duplicates()
        isoform_fasta = pd.DataFrame(columns=['uniprotID', 'isoformSequence'])
        iso_dict = []
        for i in range(len(up_list)):
            iso_dict.append(get_isoforms(up_list[i]))

        index = 0
        for i in iso_dict:
            for key, val in i.items():
                isoform_fasta.at[index, 'uniprotID'] = key
                isoform_fasta.at[index, 'isoformSequence'] = val
                index += 1
        isoform_fasta = isoform_fasta.drop_duplicates()

        for i in isoform_fasta.index:
            isoform_fasta.at[i, 'whichIsoform'] = isoform_fasta.at[i, 'uniprotID'][7:10].strip()
            isoform_fasta.at[i, 'uniprotID'] = isoform_fasta.at[i, 'uniprotID'][0:6]
        print('Sequence files created...\n')

        data = data.merge(canonical_fasta, on='uniprotID', how='left')
        data = data.astype(str)
        data['whichIsoform'] = 'nan'
        data.replace({'': 'nan'}, inplace=True)
        data['wt_sequence_match'] = ''
        for i in data.index:
            if len(data.at[i, 'uniprotSequence']) >= int(data.at[i, 'pos']):
                wt = data.at[i, 'wt']
                can = str(data.at[i, 'uniprotSequence'])[int(data.at[i, 'pos']) - 1]
                if wt == can:
                    data.at[i, 'wt_sequence_match'] = 'm'
                elif wt != can:
                    isoList = isoform_fasta[isoform_fasta['uniprotID'] == data.at[i, 'uniprotID']].isoformSequence.to_list()
                    for k in isoList:
                        if len(k) >= int(data.at[i, 'pos']):
                            resInIso = k[int(int(data.at[i, 'pos']) - 1)]
                            if wt == resInIso:
                                whichIsoform = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                                data.at[i, 'wt_sequence_match'] = 'i'
                                data.at[i, 'whichIsoform'] = whichIsoform
                                break

            elif len(data.at[i, 'uniprotSequence']) < int(data.at[i, 'pos']):
                isoList = isoform_fasta[isoform_fasta['uniprotID'] == data.at[i, 'uniprotID']].isoformSequence.to_list()
                for k in isoList:
                    if len(k) >= int(data.at[i, 'pos']):
                        resInIso = k[int(int(data.at[i, 'pos']) - 1)]
                        wt = data.at[i, 'wt']
                        if wt == resInIso:
                            whichIsoform = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                            data.at[i, 'wt_sequence_match'] = 'i'
                            data.at[i, 'whichIsoform'] = whichIsoform
                            break

        data.wt_sequence_match = data.wt_sequence_match.astype('str')
        data.replace({'': 'nan'}, inplace=True)
        data_size = len(data.drop_duplicates(['datapoint']))
        not_match_in_uniprot = data[(data.uniprotSequence == 'nan') | (data.wt_sequence_match == 'nan')]
        uniprot_matched = data[(data.uniprotSequence != 'nan') & (data.wt_sequence_match != 'nan')]
        up = uniprot_matched.at[i, 'uniprotID']
        data = None

        print('You have %d data points that failed to match a UniProt Sequence\nProceeding with %d remaining...\n'
              % (len(not_match_in_uniprot.drop_duplicates(['datapoint'])),
                 len(uniprot_matched.drop_duplicates(['datapoint']))))

        """
        STEP 5
        Retrieve related PDB sequences, extract their sequences.
        Add to the data frame.
        """
        from urllib.error import HTTPError
        pdb_fasta = pd.DataFrame(columns=['pdbID', 'chain', 'pdbSequence'])
        pdb_info = pd.DataFrame(columns=['uniprotID', 'pdbID', 'chain', 'resolution'])

        print('Retrieving PDB structures...\n')
        pdbs = []
        protein = uniprot_matched.uniprotID.to_list()
        protein = list(set(protein))

        for prot in protein:
            pdbs.append(get_pdb_ids(prot))
        if len(pdbs)>=1:
            pdbs = [item for sublist in pdbs for item in sublist]
        else:
            pdbs =[]
        print('Processing PDB structures...\n')
        if pdbs == []:
            print('No PDB structure found for the query. ')
        """
        try:
            pdbs = [j.strip('[').strip(']').strip().strip('\'').strip('\"') for j in
                    ((',').join([str(item) for item in pdbs])).split(',')]
        except IndexError:
            pdbs = []
            print('No PDB structure found for the query. ')
        """
        print('Starting PDB structures download...\n')
        pdbs = list(filter(None, pdbs))
        pdbs = (set(pdbs))
        pdbs = [i.lower() for i in pdbs]
        pdbl = PDBList()
        parser = PDBParser()
        index = 0

        try:
            shutil.rmtree('obsolete')
        except OSError as e:
            pass
        pdb_structures_path = path_to_output_files / 'log.txt'
        existing_pdb = list(Path(path_to_output_files/'pdb_structures').glob("*"))
        existing_pdb = [str(i) for i in existing_pdb]
        existing_pdb = [i.split('/')[-1].split('.')[0].lower() for i in existing_pdb]
        cnt = 0
        for search in pdbs:
            try:
                if search.lower() not in existing_pdb:
                    file = pdbl.retrieve_pdb_file(search, pdir=Path(path_to_output_files / 'pdb_structures'), file_format="pdb")
                else:
                    print('PDB structure file exists..')
                    for filename in list(Path(path_to_output_files / 'pdb_structures').glob("*")):
                        filename_replace_ext = filename.with_suffix(".pdb")
                        filename.rename(filename_replace_ext)

                    file = Path(path_to_output_files / 'pdb_structures' / f'{search}.pdb')

                    base = os.path.splitext(str(file))[0]
                    base = '/'.join(base.split('/')[0:-1]) + '/pdb' + base.split('/')[-1]
                    os.rename(file, base + ".ent")
                    file = base + '.ent'

                resolution_method = parser.get_structure(search, file)
                for record in SeqIO.parse(file, "pdb-seqres"):
                    if record.dbxrefs[0].split(':')[0] == 'UNP':
                        pdb_fasta.at[index, 'pdbID'] = record.id.split(':')[0]
                        pdb_fasta.at[index, 'chain'] = record.id.split(':')[1]
                        pdb_fasta.at[index, 'pdbSequence'] = str(record.seq)
                        pdb_info.at[index, 'uniprotID'] = record.dbxrefs[0].split(':')[1]
                        pdb_info.at[index, 'pdbID'] = record.id.split(':')[0]
                        pdb_info.at[index, 'chain'] = record.annotations["chain"]
                        pdb_info.at[index, 'resolution'] = resolution_method.header['resolution']
                    else:
                        def fetch_uniprot_ids(pdb_code):
                            response = requests.get(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{search}")

                            response.raise_for_status()  # Check for a successful response
                            data = response.json()

                            return list(list(list(data.values())[0].values())[0].keys())

                        for search in pdbs:
                            # Step 1: Fetch the PDB file
                            pdb_url = f"https://files.rcsb.org/download/{search}.pdb"

                            try:
                                response = requests.get(pdb_url)
                                response.raise_for_status()  # Check for a successful response
                            except:
                                continue  # Skip to the next PDB code if fetching fails

                            # Step 2: Parse the PDB file from memory
                            pdb_data = response.text
                            pdb_parser = PDBParser(QUIET=True)  # QUIET=True suppresses warnings
                            pdb_file_content = StringIO(pdb_data)
                            structure = pdb_parser.get_structure(search, pdb_file_content)
                            ppb = PPBuilder()
                            pdb_data_list = pdb_data.split('\n')
                            pdb_data_list_sequence = [i for i in pdb_data_list if i.startswith('SEQRES')]
                            pdb_data_list_sequence = [list(filter(None, i.split(' '))) for i in pdb_data_list_sequence]
                            seqs = {}
                            for i in pdb_data_list_sequence:
                                if i[2] in seqs.keys():
                                    seqs[i[2]] += i[4:]
                                else:
                                    seqs[i[2]] = i[4:]

                            for key, val in seqs.items():
                                seqs[key] = ''.join([threeToOne(i) for i in val])
                            pdb_data_list = [i for i in pdb_data_list if i.startswith('DBREF')]
                            pdb_data_list = [[list(filter(None, i.split(' '))) for j in i.split(' ') if j == 'UNP'] for
                                             i in pdb_data_list]
                            pdb_data_list = [i for i in pdb_data_list if i != []]
                            pdb_data_list_uniprot = [[j[6] for j in i] for i in pdb_data_list]

                            # pdb_data_list = [[list(filter(None,j)) for j in i] for i in pdb_data_list]
                            pdb_data_list = [[j[2] for j in i] for i in pdb_data_list]
                            pdb_data_list = [i[0] for i in pdb_data_list]
                            for model in structure:
                                for pp in ppb.build_peptides(model):
                                    sequence = pp.get_sequence()

                                for chain, up in zip(model, pdb_data_list_uniprot):
                                    chain_id = chain.get_id()
                                    # Extract UniProt ID if available in the chain's annotations
                                    uniprot_ids = fetch_uniprot_ids(search)
                                    # Get the resolution from the PDB header
                                    header = structure.header
                                    resolution = header.get('resolution', 'N/A')
                                    if chain_id in pdb_data_list:
                                        # Print UniProt IDs, chain ID, and resolution for the current model
                                        chain_id = chain.get_id()
                                        # st.write(f"---- Information for Chain {chain_id} in Model {i} ----")
                                        # st.write(f"UniProt IDs: {', '.join(uniprot_ids)}")
                                        # st.write(f"Chain ID: {chain_id}")
                                        # st.write(f"PDB ID: {search.upper()}")
                                        # st.write(f"Resolution: {resolution}")
                                        # st.write(f"Sequence: {sequence}")
                                        pdb_fasta.at[index, 'pdbID'] = search
                                        pdb_fasta.at[index, 'chain'] = chain_id
                                        pdb_fasta.at[index, 'pdbSequence'] = str(seqs[chain_id])
                                        pdb_info.at[index, 'uniprotID'] = ', '.join(up)
                                        pdb_info.at[index, 'pdbID'] = search
                                        pdb_info.at[index, 'chain'] = chain_id
                                        pdb_info.at[index, 'resolution'] = resolution
                                        index += 1
                    index += 1
            except:
                IndexError
                pdb_info.at[index, 'uniprotID'] = 'nan'
                pdb_info.at[index, 'pdbID'] = 'nan'
                pdb_info.at[index, 'chain'] = 'nan'
                pdb_info.at[index, 'resolution'] = 'nan'
            cnt +=1
        print()
        print('PDB file processing finished..')
        for filename in list(Path(path_to_output_files / 'pdb_structures').glob("*")):
            try:
                filename_replace_ext = filename.with_suffix(".pdb")
                filename.rename(filename_replace_ext)
            except:
                FileNotFoundError

        for filename in list(Path(path_to_output_files / 'pdb_structures').glob("*")):
            try:
                if filename.stem.startswith("pdb"):
                    filename_replace_ext = filename.with_name(filename.stem[3:])
                    filename.rename(filename_replace_ext.with_suffix('.pdb'))
            except:
                FileNotFoundError

        uniprot_matched = pd.merge(uniprot_matched, pdb_info, on='uniprotID', how='left')
        uniprot_matched = uniprot_matched.astype(str)
        uniprot_matched = uniprot_matched.drop_duplicates()

        uniprot_matched = uniprot_matched.merge(pdb_fasta, on=['pdbID', 'chain'], how='left')
        uniprot_matched = uniprot_matched.astype(str)

        with_pdb = uniprot_matched[(uniprot_matched.pdbID != 'nan') & (
                (uniprot_matched.resolution != 'nan') & (uniprot_matched.resolution != 'OT') & (
                uniprot_matched.resolution != 'None'))].drop_duplicates()
        no_pdb = uniprot_matched[(uniprot_matched.pdbID == 'nan') | (
                (uniprot_matched.resolution == 'nan') | (uniprot_matched.resolution == 'OT') | (
                uniprot_matched.resolution == 'None'))]
        no_pdb = no_pdb[~no_pdb.datapoint.isin(with_pdb.datapoint.to_list())]
        no_pdb.drop(columns=['chain', 'pdbID', 'pdbSequence', 'resolution'], inplace=True)

        print(
            'PDB Information successfully added...\nPDB structures are found for %d of %d.\n%d of %d failed to match with PDB structure.\n'
            % (len(with_pdb.drop_duplicates(['datapoint'])), len(uniprot_matched.drop_duplicates(['datapoint'])),
               len(no_pdb.drop_duplicates(['datapoint'])), len(uniprot_matched.drop_duplicates(['datapoint']))))

        with_pdb = with_pdb.sort_values(['uniprotID', 'resolution'], axis=0, ascending=True)
        with_pdb = with_pdb.drop_duplicates(['uniprotID', 'wt', 'mut', 'pos', 'pdbSequence'], keep='first')
        with_pdb.replace({'': 'nan'}, inplace=True)

        if len(with_pdb) == 0:
            with_pdb['pdbInfo'] = ''
        else:
            for i in with_pdb.index:
                try:
                    res = str(with_pdb.at[i, 'resolution'])
                    chain = with_pdb.at[i, 'chain']
                    new = with_pdb.at[i, 'pdbID'] + ':' + chain + ':' + res
                    with_pdb.at[i, 'pdbInfo'] = new
                except:
                    TypeError
                    with_pdb.at[i, 'pdbInfo'] = 'nan'

        with_pdb = with_pdb[['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                             'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence', 'pdbSequence',
                             'wt_sequence_match',
                             'whichIsoform', 'pdbID', 'resolution', 'chain', 'pdbInfo', 'datapoint']]



        # If the query data points are found in no_match_in_uniprot data frame, it will not give any results.
        # If the query data points are found in no_pdb data frame, it will be searched in the modbase and swiss_model steps.
        # If the query data points are found in with_pdb data frame, it will be searched in the following steps.

        """
        STEP 6
        Retrieve sequence annotations.
        Add to the data frame.
        """

        if len(with_pdb) > 0:
            with_pdb = add_annotations(with_pdb)
        else:
            new_cols = with_pdb.columns.to_list() + ['disulfide', 'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding',
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
            with_pdb = pd.DataFrame(columns = new_cols)
        try:
            with_pdb.whichIsoform = with_pdb.whichIsoform.astype('str')
        except:
            AttributeError
            with_pdb['whichIsoform'] = ''

        with_pdb = with_pdb.astype(str)
        with_pdb = with_pdb.replace({'NaN': 'nan'})
        with_pdb.replace({'[]': 'nan'}, inplace=True)
        with_pdb.replace({'nan-nan': 'nan'}, inplace=True)
        with_pdb.replace({'': 'nan'}, inplace=True)

        """
        STEP 7
        Do alignment for PDB
        """
        # Canonical matches, i.e. labelled as m, canonical sequences will be aligned with PDB sequences.
        # Isoform matches, i.e. labelled as i, isoform sequences will be aligned with PDB sequences.
        with_pdb['uniprotSequence'] = with_pdb['uniprotSequence'].str.replace('U', 'C')
        with_pdb['pdbSequence'] = with_pdb['pdbSequence'].str.replace('U', 'C')

        dfM = with_pdb[with_pdb.wt_sequence_match == 'm']
        dfM = dfM.sort_values(['uniprotID', 'resolution'], axis=0, ascending=True)
        dfM = dfM.drop_duplicates(['uniprotID', 'wt', 'mut', 'pos', 'pdbSequence'], keep='first')

        dfNM = with_pdb[with_pdb.wt_sequence_match == 'i']
        dfNM = dfNM.sort_values(['uniprotID', 'resolution'], axis=0, ascending=True)
        dfNM = dfNM.drop_duplicates(['uniprotID', 'wt', 'mut', 'pos', 'pdbSequence'], keep='first')
        dfNM.rename(columns={'isoformSequence': 'uniprotSequence'}, inplace=True)

        dfM = dfM.astype(str)
        dfNM = dfNM.astype(str)

        dfM.reset_index(inplace=True)
        dfM.drop(['index'], axis=1, inplace=True)
        dfNM.reset_index(inplace=True)
        dfNM.drop(['index'], axis=1, inplace=True)

        uniprot_matched_size = len(uniprot_matched.drop_duplicates(['datapoint']))
        uniprot_matched = None
        pdb_fasta = None
        pdb_info = None
        pdbs = None
        existing_pdb = None
        with_pdb_size = len(with_pdb.drop_duplicates(['datapoint']))
        with_pdb = None

        print('Aligning sequences...\n')
        aligned_m = final_stage(dfM, annotation_list, Path(path_to_output_files / 'alignment_files'))
        aligned_nm = final_stage(dfNM, annotation_list, Path(path_to_output_files / 'alignment_files'))

        # When PDB sequence is nan, it is wrongly aligned to the UniProt sequence. Fix them.
        for i in aligned_m.index:
            if aligned_m.at[i, 'pdbSequence'] == 'nan':
                aligned_m.at[i, 'mutationPositionOnPDB'] = 'nan'
                aligned_m.at[i, 'domainStartonPDB'] = 'nan'
                aligned_m.at[i, 'domainEndonPDB'] = 'nan'
                aligned_m.at[i, 'pdb_alignStatus'] = 'nan'

        for i in aligned_nm.index:
            if aligned_nm.at[i, 'pdbSequence'] == 'nan':
                aligned_nm.at[i, 'mutationPositionOnPDB'] = 'nan'
                aligned_nm.at[i, 'domainStartonPDB'] = 'nan'
                aligned_nm.at[i, 'domainEndonPDB'] = 'nan'
                aligned_nm.at[i, 'pdb_alignStatus'] = 'nan'

        # Check if they the same column name before merging.
        aligned_m = aligned_m.astype(str)
        aligned_nm = aligned_nm.astype(str)


        frames = [aligned_m, aligned_nm]
        after_up_pdb_alignment = pd.concat(frames, sort=False)
        if len(after_up_pdb_alignment) == 0:
            after_up_pdb_alignment['pdb_alignStatus'] = ''
            after_up_pdb_alignment['mutationPositionOnPDB'] = ''
            after_up_pdb_alignment['domainStartonPDB'] = ''
            after_up_pdb_alignment['domainEndonPDB'] = ''

        after_up_pdb_alignment = after_up_pdb_alignment.sort_values(
            by=['uniprotID', 'wt', 'mut', 'pos', 'pdb_alignStatus', 'resolution', 'chain'],
            ascending=[True, True, True, True, True, True, True])

        after_up_pdb_alignment = after_up_pdb_alignment.drop_duplicates(['uniprotID', 'wt', 'mut', 'pos'], keep='first')

        after_up_pdb_alignment = after_up_pdb_alignment.astype('str')

        pdb_aligned = after_up_pdb_alignment[
            (after_up_pdb_alignment.pdbID != 'nan') & (after_up_pdb_alignment.mutationPositionOnPDB != 'nan')]
        yes_pdb_no_match = after_up_pdb_alignment[
            (after_up_pdb_alignment.pdbID != 'nan') & (after_up_pdb_alignment.mutationPositionOnPDB == 'nan')]
        no_pdb = no_pdb.copy()


        print('PDB matching is completed...\n')
        print('SUMMARY')
        print('-------')
        print('%d data points that failed to match a UniProt Sequence are discarded.' % len(
            not_match_in_uniprot.drop_duplicates(['datapoint'])))
        print('Of the remaining %d:' % uniprot_matched_size)
        print('--%d of %d successfully aligned with PDB structures.' % (
            len(pdb_aligned.drop_duplicates(['datapoint'])), with_pdb_size))
        print('--%d of %d not found on the covered area by the structure.' % (
            len(yes_pdb_no_match.drop_duplicates(['datapoint'])), with_pdb_size))
        print('--PDB structures not found for %d datapoints.' % len(no_pdb.drop_duplicates(['datapoint'])))
        print('--%d will be searched in Swiss-Model database.\n' % (
                len(yes_pdb_no_match.drop_duplicates(['datapoint'])) + len(no_pdb.drop_duplicates(['datapoint']))))


        dfM = None
        dfNM = None
        aligned_nm = None
        aligned_m = None
        after_up_pdb_alignment = None

        print('Proceeding to  SwissModel search...')
        print('------------------------------------\n')

        # At this point we have 4 dataframes
        # 1. after_up_pdb_alignment --- This is after PDB sequence alignment. There may be mutations that wasnt found matching to after the alignment. Will be searched in other databases as well.
        # 1a. aligned --- we are done with this.
        # 1b. yes_pdb_no_match --- They have PDB structures but not matched, so will be searched in the other databases.
        # 2. not_match_in_uniprot --- This wont be aligned with anything because these proteins dont have a uniprot ID. Only basic info is present.
        # 3. no_pdb --- No PDB structures were found for them. Will be searched in other databases.

        """
        Step 8
        Neutralize data points that are to be searched in Swiss-Model
        # One point is that yes_pdb_no_match's annotations are the adjusted according to the PDBs they are matched before.
        # They need to be converted to their old original UniProt annotation positions.
        """
        yes_pdb_no_match.drop(['disulfide', 'intMet',
                               'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                               'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink',
                               'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                               'caBinding', 'topologicalDomain', 'bindingSite', 'region',
                               'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil',
                               'peptide', 'transitPeptide', 'glycosylation', 'propeptide', 'disulfideBinary',
                               'intMetBinary', 'intramembraneBinary',
                               'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary',
                               'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
                               'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
                               'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                               'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary',
                               'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
                               'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
                               'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
                               'glycosylationBinary', 'propeptideBinary', 'pdbSequence', 'pdbInfo', 'pdbID',
                               'chain', 'resolution', 'pdb_alignStatus', 'mutationPositionOnPDB',
                               'domainStartonPDB', 'domainEndonPDB'], axis=1, inplace=True)

        to_swiss = pd.concat([yes_pdb_no_match.drop_duplicates(['datapoint']), no_pdb.drop_duplicates(['datapoint'])])
        no_pdb = None
        to_swiss.reset_index(inplace=True)
        to_swiss.drop(['index'], axis=1, inplace=True)
        to_swiss = to_swiss.astype('str')
        to_swiss = to_swiss.replace({'NaN': 'nan'})
        # Create model summary dataframe.
        if len(to_swiss) != 0:
            print('Generating SwissModel file...\n')

            swiss_model = pd.read_csv(Path(path_to_input_files / 'swissmodel_structures.txt'), sep='\t',
                                      dtype=str, header=None, skiprows=1,
                                      names=['UniProtKB_ac', 'iso_id', 'uniprot_seq_length', 'uniprot_seq_md5',
                                             'coordinate_id', 'provider', 'from', 'to', 'template', 'qmean', 'qmean_norm','seqid', 'url'])

        else:
            swiss_model = pd.DataFrame(
                columns=['UniProtKB_ac', 'iso_id', 'uniprot_seq_length', 'uniprot_seq_md5', 'coordinate_id',
                         'provider', 'from', 'to', 'template', 'qmean', 'qmean_norm', 'seqid', 'url', 'whichIsoform'])
        swiss_model = swiss_model.astype('str')
        try:
            swiss_model.iso_id = swiss_model.iso_id.astype('str')
        except:
            AttributeError
            swiss_model['iso_id'] = 'nan'
        swiss_model = swiss_model[swiss_model.UniProtKB_ac != 'nan']
        for ind in swiss_model.index:
            swiss_model.at[ind, 'UniProtKB_ac'] = swiss_model.at[ind, 'UniProtKB_ac'].split('-')[0]
            if swiss_model.at[ind, 'iso_id'] != 'nan':

                swiss_model.at[ind, 'whichIsoform'] = swiss_model.at[ind, 'iso_id'].split('-')[1]
            else:
                swiss_model.at[ind, 'whichIsoform'] = 'nan'
#        swiss_model.drop(['input'], axis=1, inplace=True)
        swiss_model = swiss_model[swiss_model.provider == 'SWISSMODEL']
        print('Index File Processed...\n')


        # Get relevant columns
        swiss_model = swiss_model[['UniProtKB_ac', 'from', 'to', 'template', 'qmean_norm', 'seqid', 'url', 'whichIsoform']]
        # Sort models on qmean score and identity. Some proteins have more than one models, we will pick one.
        swiss_model = swiss_model.sort_values(by=['UniProtKB_ac', 'qmean_norm', 'seqid'], ascending=False)
        swiss_model.reset_index(inplace=True)
        swiss_model.drop(['index'], axis=1, inplace=True)

        # Get protein IDs for which there exist models.
        swiss_model_ids = set(swiss_model.UniProtKB_ac.to_list())
        to_swiss = to_swiss.astype(str)
        no_swiss_models = pd.DataFrame()
        for i in to_swiss.index:
            if to_swiss.at[i, 'uniprotID'] not in swiss_model_ids:
                k = pd.Series(to_swiss.iloc[i])
                no_swiss_models = no_swiss_models.append(k, ignore_index=True)

        no_swiss_models = no_swiss_models.astype(str)
        if len(no_swiss_models) == 0:
            no_swiss_models = pd.DataFrame(columns=to_swiss.columns)
        else:
            no_swiss_models = no_swiss_models[to_swiss.columns]
            no_swiss_models.reset_index(inplace=True)
            no_swiss_models.drop('index', axis=1, inplace=True)

        with_swiss_models = pd.concat([to_swiss, no_swiss_models]).drop_duplicates(['datapoint'], keep=False)
        with_swiss_models = with_swiss_models[to_swiss.columns]

        # Add model info.

        with_swiss_models = with_swiss_models.astype(str)
        swiss_model = swiss_model.astype(str)
        swiss_models_with_data = pd.merge(with_swiss_models, swiss_model, left_on=['uniprotID', 'whichIsoform'],
                                          right_on=['UniProtKB_ac', 'whichIsoform'],
                                          how='left')
        swiss_models_with_data = swiss_models_with_data.astype(str)
        swiss_models_with_data = swiss_models_with_data.sort_values(by=['uniprotID', 'wt', 'mut', 'pos', 'qmean_norm'],
                                                                    ascending=False)
        swiss_models_with_data = swiss_models_with_data.drop_duplicates()
        swiss_models_with_data = swiss_models_with_data.drop(['UniProtKB_ac', 'seqid'], axis=1)
        swiss_models_with_data.pos = swiss_models_with_data.pos.astype('int')
        swiss_models_with_data = swiss_models_with_data.astype(str)

        # Get the ones in the list but without model url and add to the list to go to modbase.
        url_nan = swiss_models_with_data[swiss_models_with_data.url == 'nan']

        # Add this nan's to no_model. These will be searched in MODBASE because here they dont have urls.
        url_nan = url_nan.drop(['from', 'qmean_norm', 'template', 'to', 'url'], axis=1)

        no_swiss_models_2 = pd.concat([no_swiss_models, url_nan])
        swiss_models_with_data = swiss_models_with_data[swiss_models_with_data.url != 'nan']
        for i in swiss_models_with_data.index:
            try:
                swiss_models_with_data.at[i, 'chain'] = swiss_models_with_data.at[i, 'template'].split('.')[2]
                swiss_models_with_data.at[i, 'template'] = swiss_models_with_data.at[i, 'template'].split('.')[0]
            except:
                IndexError
        if len(swiss_models_with_data) == 0:
            swiss_models_with_data['chain'] = ''
            swiss_models_with_data['template'] = ''

        swiss_models_with_data.qmean_norm = swiss_models_with_data.qmean_norm.astype('str')
        swiss_models_with_data.chain = swiss_models_with_data.chain.astype('str')
        swiss_models_with_data['qmean_norm'] = swiss_models_with_data.qmean_norm.apply(lambda x: round(float(x), 2))
        swiss_models_with_data = swiss_models_with_data.astype(str)

        # swiss_models_with_data: These data points will be aligned with their corresponding model sequences.
        # Add sequences

        no_swiss_models_2.reset_index(inplace=True)
        no_swiss_models_2.drop('index', axis=1, inplace=True)

        swiss_models_with_data.reset_index(inplace=True)
        swiss_models_with_data.drop('index', axis=1, inplace=True)

        swiss_model_ids = None
        with_swiss_models = None
        swiss_model = None
        no_swiss_models = None
        url_nan = None

        # At this point we have:
        # pdb_aligned --- Align in the PDB phase
        # not_match_in_uniprot --- This wont be aligned with anything because these proteins dont have a uniprot ID. Only basic info is present.
        # to_swiss (no_pdb + yes_pdb_no_match) --- to be searched in SwissModel database
        # to_swiss (with_swiss_models & no_swiss_models)
        # swiss_models_with_data --- We found swiss models for them.
        # no_swiss_models_2 (no_swiss_models + url_nan)--- to be searched in modbase (the ones having swissmodels but not matching with the boundaries  & broken_swiss will be added here)

        """
        STEP 9
        Associated model IDs are added. 
        Download model files.
        """
        print('Beginning SwissModel files download...')
        existing_swiss = list(Path(path_to_output_files / 'swissmodel_structures').glob("*"))
        existing_swiss = [str(i) for i in existing_swiss]
        existing_swiss = ['.'.join(i.split('/')[-1].split('.')[:-1]) for i in existing_swiss]
        swissmodels_fasta = pd.DataFrame()

        for i in swiss_models_with_data.index:
            protein = swiss_models_with_data.at[i, 'uniprotID']
            template = swiss_models_with_data.at[i, 'template'].split('.')[0]
            qmean_norm = str(round(float(swiss_models_with_data.at[i, 'qmean_norm']), 2))
            if protein + '_' + template + '_' + qmean_norm not in existing_swiss:
                url = swiss_models_with_data.at[i, 'url'].strip('\"').strip('}').replace('\\', '').strip('\"').replace(
                    'https',
                    'https:')
                req = requests.get(url)
                name = Path(path_to_output_files / 'swissmodel_structures' / f'{protein}_{template}_{qmean_norm}.txt')
                print('Downloading for Protein:', protein + ' Model: ' + template)
                with open(name, 'wb') as f:
                    f.write(req.content)
            else:
                print('Model exists.')
                name = Path(path_to_output_files / 'swissmodel_structures' / f'{protein}_{template}_{qmean_norm}.txt')
            with open(name, encoding="utf8") as f:
                fasta = ''
                lines = f.readlines()
                chain = ''
                for row in lines:
                    if row[0:4] == 'ATOM' and row[13:15] == 'CA':
                        chain = row[20:22].strip()
                        fasta += threeToOne(row[17:20])
                    if row[0:3] == 'TER':
                        k = pd.Series([protein, template, qmean_norm, chain.upper(), fasta])
                        swissmodels_fasta = swissmodels_fasta.append(k, ignore_index=True)
                        fasta = ''

        if len(swissmodels_fasta) == 0:
            swissmodels_fasta = pd.DataFrame(columns=['uniprotID', 'template', 'qmean_norm', 'chain', 'fasta'])
        else:
            swissmodels_fasta.columns = ['uniprotID', 'template', 'qmean_norm', 'chain', 'fasta']

        swissmodels_fasta = swissmodels_fasta.astype(str)

        swiss_models_with_data.qmean_norm = swiss_models_with_data.qmean_norm.astype(float)
        swissmodels_fasta.qmean_norm = swissmodels_fasta.qmean_norm.astype(float)

        swissmodels_fasta = swissmodels_fasta.sort_values(['uniprotID', 'template', 'qmean_norm', 'chain'],
                                                          axis=0)  # example = 3gdh
        swissmodels_fasta.reset_index(inplace=True)
        swissmodels_fasta.drop(['index'], axis=1, inplace=True)
        swissmodels_fasta = swissmodels_fasta.drop_duplicates(['uniprotID', 'template', 'qmean_norm', 'chain'])
        swissmodels_fasta = swissmodels_fasta.drop_duplicates(['uniprotID', 'template', 'chain', 'fasta'])
        swissmodels_fasta = swissmodels_fasta.drop_duplicates(['uniprotID', 'template', 'fasta'])
        # Some files were broken, thus their PDBs couldnt be recorded.
        swissmodels_fasta = swissmodels_fasta.drop_duplicates()
        swissmodels_fasta = swissmodels_fasta.astype(str)

        swiss_models_with_data = swiss_models_with_data.astype(str)
        swissmodels_fasta = swissmodels_fasta.astype(str)
        swiss_models_with_data1 = swiss_models_with_data.merge(swissmodels_fasta,
                                                               on=['uniprotID', 'template', 'qmean_norm', 'chain'])

        swiss_models_with_data1 = swiss_models_with_data1.sort_values(['datapoint', 'fasta'], axis=0,
                                                                      ascending=[True, False])
        swiss_models_with_data1 = swiss_models_with_data1.drop_duplicates(['datapoint', 'template'])


        swiss_models_with_data1_dp = list(set(swiss_models_with_data1.datapoint.to_list()))
        swiss_models_with_data.reset_index(inplace=True)
        swiss_models_with_data.drop(['index'], axis=1, inplace=True)
        broken_swiss = pd.DataFrame()
        c = 0
        for i in swiss_models_with_data.index:  # en baştaki dfde var ama model gelende yok.
            if swiss_models_with_data.at[i, 'datapoint'] not in swiss_models_with_data1_dp:
                k = pd.Series(swiss_models_with_data.iloc[i])
                broken_swiss = broken_swiss.append(k, ignore_index=True)
                c += 1

        if len(broken_swiss) == 0:
            broken_swiss = pd.DataFrame(columns=swiss_models_with_data.columns.to_list())

        swiss_models_with_data = swiss_models_with_data1.copy()


        swiss_models_with_data.qmean_norm = swiss_models_with_data.qmean_norm.astype('float')
        swiss_models_with_data = swiss_models_with_data.sort_values(['uniprotID', 'wt', 'mut', 'qmean_norm'],
                                                                    axis=0, ascending=[True, True, True, False])

        # Delete the same model sequence with lower quality
        swiss_models_with_data = swiss_models_with_data.drop_duplicates(['uniprotID', 'wt', 'mut', 'pos', 'fasta'],
                                                                        keep='first')
        swiss_models_with_data.uniprotSequence = swiss_models_with_data.uniprotSequence.astype('str')
        swiss_models_with_data.pos = swiss_models_with_data.pos.astype('int')
        len(swiss_models_with_data.drop_duplicates(['datapoint'])) + len(broken_swiss.drop_duplicates(['datapoint'])) + len(
            no_swiss_models_2.drop_duplicates(['datapoint'])) == len(to_swiss.drop_duplicates(['datapoint']))
        # This printed data here includes all possible models with different qualities,
        # because we may get a hit in either of them.
        swiss_models_with_data.rename({'fasta': 'pdbSequence'}, axis=1, inplace=True)  # for convenience.

        # NOW DO ALIGNMENT HERE

        swiss_models_with_data = swiss_models_with_data.replace({'[\'?\']': 'nan'})
        swiss_models_with_data = swiss_models_with_data.replace({'[]': 'nan'})
        swiss_models_with_data.rename({'template': 'pdbID'}, axis=1,
                                      inplace=True)  # Only to be able use the alignment code above.
        swiss_models_with_data = swiss_models_with_data.astype(str)
        swiss_models_with_data.pdbSequence = swiss_models_with_data.pdbSequence.astype('str')
        swiss_models_with_data = add_annotations(swiss_models_with_data)
        swiss_models_with_data = swiss_models_with_data.astype(str)
        swiss_models_with_data.replace({'NaN': 'nan'}, inplace=True)
        swiss_models_with_data_copy = swiss_models_with_data.copy()
        swiss_models_with_data1_dp = None
        swiss_models_with_data1 = None
        existing_swiss = None
        swissmodels_fasta = None

        print('Aligning sequences...\n')

        swiss_models_with_data['uniprotSequence'] = swiss_models_with_data['uniprotSequence'].str.replace('U', 'C')
        swiss_models_with_data['pdbSequence'] = swiss_models_with_data['pdbSequence'].str.replace('U', 'C')
        swiss_model_aligned = alignment(swiss_models_with_data, annotation_list, path_to_output_files / 'alignment_files')
        swiss_models_with_data = None


        if len(swiss_model_aligned) == 0:
            swiss_model_aligned = pd.DataFrame(columns=pdb_aligned.columns)
            swiss_model_aligned['qmean_norm'] = 'nan'
        else:
            swiss_model_aligned = swiss_model_aligned.astype(str)
            swiss_model_aligned.replace({'NaN': 'nan'}, inplace=True)

        # Some datapoints appear in both nan and not_nan. If not_nan we take it only once.
        nan = swiss_model_aligned[swiss_model_aligned.mutationPositionOnPDB == 'nan']
        not_nan = swiss_model_aligned[swiss_model_aligned.mutationPositionOnPDB != 'nan']
        not_nan.qmean_norm = not_nan.qmean_norm.astype('float')
        not_nan.sort_values(['datapoint', 'pdb_alignStatus', 'qmean_norm'], ascending=[True, True, False], inplace=True)

        which_ones_are_match = pd.concat([not_nan, nan]).drop_duplicates(['datapoint'], keep='first')
        swiss_match = which_ones_are_match[which_ones_are_match.mutationPositionOnPDB != 'nan']
        swiss_not_match = which_ones_are_match[which_ones_are_match.mutationPositionOnPDB == 'nan']

        swiss_match.qmean_norm = swiss_match.qmean_norm.astype('float')
        swiss_match.sort_values(['uniprotID', 'wt', 'pos', 'mut', 'pdb_alignStatus', 'qmean_norm'],
                                ascending=[True, True, True, True, True, False], inplace=True)
        swiss_match.drop_duplicates(['uniprotID', 'wt', 'pos', 'mut'], keep='first', inplace=True)
        swiss_not_match = swiss_not_match[no_swiss_models_2.columns]
        broken_swiss = broken_swiss[no_swiss_models_2.columns]
        swiss_not_match = swiss_not_match.drop_duplicates(['datapoint'])
        broken_swiss = broken_swiss.drop_duplicates(['datapoint'])

        to_modbase = pd.concat([no_swiss_models_2, broken_swiss]).drop_duplicates()
        to_modbase = pd.concat([to_modbase, swiss_not_match]).drop_duplicates()
        to_modbase = to_modbase.astype(str)
        to_swiss_columns = to_swiss.columns
        to_swiss_size = len(to_swiss.drop_duplicates(['datapoint']))
        to_swiss = None

        # CONTROL

        """
        # This should be the whole data.
        len(swiss_match.drop_duplicates(['datapoint'])) + len(aligned.drop_duplicates(['datapoint'])) + len(to_modbase.drop_duplicates(['datapoint'])) + len(not_match_in_uniprot.drop_duplicates(['datapoint'])) ,len(data)
        len(aligned.drop_duplicates(['datapoint'])) + len(not_match_in_uniprot.drop_duplicates(['datapoint'])) +len(to_swiss.drop_duplicates(['datapoint']))== len(data)
        """
        print('SwissModel matching is completed...\n')
        print('SUMMARY')
        print('-------')
        print('%d data points that failed to match a UniProt Sequence are discarded.' % len(
            not_match_in_uniprot.drop_duplicates(['datapoint'])))
        print('Of the remaining %d:' % uniprot_matched_size)
        print('--%d of %d successfully aligned with PDB structures.' % (
            len(pdb_aligned.drop_duplicates(['datapoint'])), with_pdb_size))
        print('--%d of %d successfully aligned with SwissModels structures.' % (
            len(swiss_match.drop_duplicates(['datapoint'])), to_swiss_size))
        print('--%d will be searched in ModBase database.\n' % len(to_modbase.drop_duplicates(['datapoint'])))

        print('Proceeding to ModBase search...')
        print('------------------------------------\n')
        no_swiss_models_2 = None
        broken_swiss = None
        swiss_model_aligned = None
        nan = None
        not_nan = None
        which_ones_are_match = None
        swiss_not_match = None

        # STEP :  GO TO MODBASE
        # Should not include anything related to prev models.
        if len(to_modbase) != 0:
            to_modbase = to_modbase.astype(str)

            # GET MODBASE MODELS

            # Get IDs from data to retrieve only their models from MODBASE
            to_modbase.reset_index(inplace=True)
            to_modbase.drop(['index'], axis=1, inplace=True)

            existing_modbase_models = list(Path(path_to_output_files / 'modbase_structures').glob("*"))
            existing_modbase_models = [str(i) for i in existing_modbase_models]
            existing_modbase_models = [i.split('/')[-1].split('.')[0] for i in existing_modbase_models]

            existing_modbase_models_ind = list(Path(path_to_output_files / 'modbase_structures_individual').glob("*"))
            existing_modbase_models_ind = [str(i) for i in existing_modbase_models_ind]
            existing_modbase_models_ind = [i.split('/')[-1].split('.')[0] for i in existing_modbase_models_ind]

            modbase_reduced = pd.DataFrame()
            modbase_fasta = pd.DataFrame()

            print('Retrieving ModBase models...\n')
            # Get model files associated with each UniProtID
            for protein in list(set(to_modbase.uniprotID.to_list())):
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
                    for pdb in soup.findAll('pdbfile'):
                        model_id = str(pdb.contents[1])[10:-11]
                        if model_id not in existing_modbase_models_ind:
                            with open(path_to_output_files / 'modbase_structures_individual' / f'{model_id}.txt', 'w',
                                      encoding="utf8") as individual:
                                individual.write(str('UniProt ID: ' + protein))
                                individual.write('\n')
                                individual.write(str(pdb.contents[3])[10:-11].strip())
                        with open(path_to_output_files / 'modbase_structures_individual'/ f'{model_id}.txt',
                                  encoding="utf8") as f:
                            fasta = ''
                            chain = ''
                            template_chain = ''
                            score = -999
                            for ind_line in f.readlines():
                                if ind_line[0:10] == 'UniProt ID':
                                    uniprot_id = ind_line.split(':')[1].strip()
                                if ind_line[0:23] == 'REMARK 220 TARGET BEGIN':
                                    target_begin = ind_line[40:43].strip()
                                if ind_line[0:21] == 'REMARK 220 TARGET END':
                                    target_end = ind_line[40:43].strip()
                                if ind_line[0:25] == 'REMARK 220 TEMPLATE BEGIN':
                                    pdb_begin = ind_line[40:43].strip()
                                if ind_line[0:23] == 'REMARK 220 TEMPLATE END':
                                    pdb_end = ind_line[40:43].strip()
                                if ind_line[0:23] == 'REMARK 220 TEMPLATE PDB':
                                    pdb_code = ind_line[40:43].strip()
                                if ind_line[0:25] == 'REMARK 220 TEMPLATE CHAIN':
                                    pdb_chain = ind_line[40:43].strip()
                                if ind_line[0:32] == 'REMARK 220 ModPipe Quality Score':
                                    quality_score = ind_line[40:].strip()
                                if ind_line[0:27] == 'REMARK 220 MODPIPE MODEL ID':
                                    model_id = ind_line[40:].strip()
                                if ind_line[0:25] == 'REMARK 220 TEMPLATE CHAIN':
                                    template_chain = ind_line[40:42].strip()
                                if ind_line[0:4] == 'ATOM' and ind_line[13:15] == 'CA':
                                    fasta += threeToOne(ind_line[17:20])
                                if ind_line[0:32] == 'REMARK 220 ModPipe Quality Score':
                                    try:
                                        score = ind_line[40:].strip()
                                    except (ValueError):
                                        score = -999
                                if ind_line[0:3] == 'TER' or ind_line[0:3] == 'END':
                                    k = pd.Series([uniprot_id, model_id, str(score), template_chain, fasta])
                                    modbase_fasta = modbase_fasta.append(k, ignore_index=True)
                                    fasta = ''
                            try:
                                k = pd.Series(
                                    [uniprot_id, target_begin, target_end, pdb_code, pdb_chain, pdb_begin, pdb_end,
                                     quality_score,
                                     model_id])
                                modbase_reduced = modbase_reduced.append(k, ignore_index=True)
                            except:
                                NameError
                                print('This file doesnt have Quality Score. Replacer: -999', model_id)
                                quality_score = -999

            print()
            if len(modbase_fasta) != 0:
                modbase_fasta.columns = ['uniprotID', 'template', 'score', 'chain', 'fasta']
            else:
                modbase_fasta = pd.DataFrame(columns=['uniprotID', 'template', 'score', 'chain', 'fasta'])
            modbase_fasta = modbase_fasta.astype(str)
            modbase_fasta = modbase_fasta.replace({'': 'nan'})
            modbase_fasta = modbase_fasta.replace({'NaN': 'nan'})
            modbase_fasta = modbase_fasta[modbase_fasta.fasta != 'nan']

            print('Modbase model frame constructed.\n')
            if len(modbase_reduced) != 0:
                modbase_reduced.columns = ['UniprotID', 'TargetBeg', 'TargetEnd', 'PDBCode', 'PDBChain', 'PDBBegin',
                                           'PDBEnd',
                                           'ModPipeQualityScore', 'ModelID']
            else:
                modbase_reduced = pd.DataFrame(
                    columns=['UniprotID', 'TargetBeg', 'TargetEnd', 'PDBCode', 'PDBChain', 'PDBBegin', 'PDBEnd',
                             'ModPipeQualityScore', 'ModelID'])

            to_modbase = add_annotations(to_modbase)

            to_modbase = to_modbase.astype(str)
            to_modbase.fillna('nan', inplace=True)
            to_modbase = to_modbase.replace({'NaN': 'nan'})
            to_modbase.replace({'[]': 'nan'}, inplace=True)
            to_modbase.replace({'nan-nan': 'nan'}, inplace=True)
            to_modbase.replace({'': 'nan'}, inplace=True)
            model_info_added = to_modbase.merge(modbase_reduced, right_on='UniprotID', left_on='uniprotID',
                                                how='left')
            modbase_reduced = None
            existing_modbase_models = None
            existing_modbase_models_ind = None


            model_info_added = model_info_added.drop(['UniprotID'], axis=1)
            model_info_added = model_info_added.rename(columns={'TargetBeg': 'from', 'TargetEnd': 'to',
                                                                'PDBCode': 'template', 'PDBChain': 'chain',
                                                                'ModPipeQualityScore': 'score',
                                                                'ModelID': 'pdbID'})
            model_info_added.drop(['PDBEnd', 'PDBBegin'], axis=1, inplace=True)
            model_info_added.score = model_info_added.score.astype(float)
            model_info_added = model_info_added.sort_values(by=['datapoint', 'score'],
                                                            ascending=False)
            model_info_added.reset_index(inplace=True)
            model_info_added.drop(['index'], axis=1, inplace=True)
            model_info_added = model_info_added.drop_duplicates()

            model_info_added = model_info_added.astype(str)
            model_info_added = model_info_added.replace({'NaN': 'nan'})
            no_info = model_info_added[model_info_added.pdbID == 'nan']
            with_modbase_info = model_info_added[model_info_added.pdbID != 'nan']
            model_info_added = None

            len(no_info.drop_duplicates(['datapoint'])), len(with_modbase_info.drop_duplicates(['datapoint']))
            len(no_info.drop_duplicates(['datapoint'])) + len(with_modbase_info.drop_duplicates(['datapoint'])) == len(
                to_modbase.drop_duplicates(['datapoint']))

            # Add no_info to the rest down below!
            no_info = no_info[to_swiss_columns]

            with_modbase_info.score = with_modbase_info.score.astype(float)
            modbase_fasta.score = modbase_fasta.score.astype(float)

            modbase_fasta = modbase_fasta.sort_values(['uniprotID', 'score', 'template', 'chain'],
                                                      ascending=[True, False, True, True], axis=0)  # example = 3gdh

            # I added this newly downloaded ones to the main model file.

            modbase_fasta = modbase_fasta.rename(columns={'template': 'pdbID'})
            with_modbase_info.pos = with_modbase_info.pos.astype('int')
            with_modbase_info.score = with_modbase_info.score.astype(float)
            with_modbase_info.score = with_modbase_info.score.apply(lambda x: round(x, 2))
            modbase_fasta.score = modbase_fasta.score.astype(float)
            modbase_fasta.score = modbase_fasta.score.apply(lambda x: round(x, 2))

            with_modbase_info = with_modbase_info.merge(modbase_fasta, on='pdbID', how='left')

            with_modbase_info.drop(['score_y'], axis=1, inplace=True)
            with_modbase_info.rename(columns={'score_x': 'score'}, inplace=True)
            with_modbase_info.drop(['uniprotID_y', 'chain_y'], axis=1, inplace=True)
            with_modbase_info.rename(columns={'uniprotID_x': 'uniprotID', 'chain_x': 'chain'}, inplace=True)

            with_modbase_info.score = with_modbase_info.score.astype('float')
            with_modbase_info = with_modbase_info.sort_values(['uniprotID', 'wt', 'mut', 'pos', 'score', 'from', 'to'],
                                                              axis=0,
                                                              ascending=[True, True, True, True, False, True, False])
            with_modbase_info = with_modbase_info.drop_duplicates(['uniprotID', 'wt', 'mut', 'pos', 'fasta'], keep='first')

            with_modbase_info = with_modbase_info.replace({'[\'?\']': 'nan'})
            with_modbase_info = with_modbase_info.replace({'[]': 'nan'})
            with_modbase_info = with_modbase_info.replace({'\'?\', ': ''})
            with_modbase_info = with_modbase_info.replace({', \'?\'': ''})
            with_modbase_info = with_modbase_info.replace({'(': ''})
            with_modbase_info = with_modbase_info.replace(
                {')': ''})
            with_modbase_info = with_modbase_info.astype(str)
            with_modbase_info.fasta = with_modbase_info.fasta.astype('str')
            with_modbase_info.reset_index(inplace=True)
            with_modbase_info.drop('index', axis=1, inplace=True)


            align = with_modbase_info[
                with_modbase_info.fasta != 'nan']
            yes_pdb_no_match = with_modbase_info[
                with_modbase_info.fasta == 'nan']
            yes_pdb_no_match = yes_pdb_no_match[~yes_pdb_no_match.datapoint.isin(align.datapoint.to_list())]

            align.rename(columns={'fasta': 'pdbSequence'}, inplace=True)
            align['uniprotSequence'] = align['uniprotSequence'].str.replace('U', 'C')
            align['pdbSequence'] = align['pdbSequence'].str.replace('U', 'C')

            to_modbase_size = len(to_modbase.drop_duplicates(['datapoint']))
            modbase_fasta = None
            to_modbase = None
            print('Aligning sequences...\n')
            modbase_aligned = alignment(align, annotation_list, path_to_output_files / 'alignment_files')
            modbase_aligned = modbase_aligned.astype(str)
            modbase_aligned = modbase_aligned.replace({'NaN': 'nan'})


            # Get the ones whose models couldn't be found. Add to no_modbase (yani hiçbir şey de eşleşmemiş artık.)
            if len(with_modbase_info) != 0:
                not_in_aligned = pd.concat([modbase_aligned.drop_duplicates(['datapoint']),
                                            with_modbase_info.drop_duplicates(['datapoint'])]).drop_duplicates(
                    ['datapoint'],
                    keep=False)
            else:
                not_in_aligned = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                                                       'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                                       'wt_sequence_match', 'whichIsoform', 'datapoint', 'disulfide',
                                                       'intMet',
                                                       'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                                                       'nucleotideBinding', 'lipidation', 'site', 'transmembrane',
                                                       'crosslink',
                                                       'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                                                       'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                                                       'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif',
                                                       'coiledCoil',
                                                       'peptide', 'transitPeptide', 'glycosylation', 'propeptide',
                                                       'disulfide',
                                                       'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding',
                                                       'activeSite',
                                                       'nucleotideBinding', 'lipidation', 'site', 'transmembrane',
                                                       'crosslink',
                                                       'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                                                       'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                                                       'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif',
                                                       'coiledCoil',
                                                       'peptide', 'transitPeptide', 'glycosylation', 'propeptide', 'from',
                                                       'to', 'template', 'chain', 'score', 'pdbID', 'pdbSequence', 'fasta'])
            with_modbase_info = None
            if len(not_in_aligned) != 0:
                not_models = pd.concat([yes_pdb_no_match.drop_duplicates(['datapoint']),
                                        not_in_aligned.drop_duplicates(['datapoint'])]).drop_duplicates(['datapoint'],
                                                                                                        keep='first')
            # Retain the best model among the aligned ones.
            else:
                not_models = pd.DataFrame(columns=not_in_aligned.columns)

            yes_pdb_no_match = None
            # # Some datapoints appear in both nan and not_nan. If not_nan we take it only once.
            modbase_aligned = modbase_aligned.astype(str)
            if len(modbase_aligned) != 0:
                nan = modbase_aligned[modbase_aligned.mutationPositionOnPDB == 'nan']
                not_nan = modbase_aligned[modbase_aligned.mutationPositionOnPDB != 'nan']
                not_nan.score = not_nan.score.astype(float)
                not_nan.sort_values(['datapoint', 'pdb_alignStatus', 'score'], ascending=[True, True, False], inplace=True)

                not_nan = not_nan.sort_values(['datapoint', 'mutationPositionOnPDB', 'score'],
                                              ascending=[True, True, False])
                not_nan = not_nan.drop_duplicates(['datapoint'], keep='first')
            else:
                nan = pd.DataFrame(columns=modbase_aligned.columns)
                not_nan = pd.DataFrame(columns=modbase_aligned.columns)
            modbase_aligned = None
            which_ones_are_match = pd.concat([not_nan, nan]).drop_duplicates(['datapoint'], keep='first')
            if len(which_ones_are_match) == 0:
                which_ones_are_match = pd.DataFrame(
                    columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                             'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                             'wt_sequence_match', 'whichIsoform', 'datapoint', 'disulfide', 'intMet',
                             'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                             'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink',
                             'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                             'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                             'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil',
                             'peptide', 'transitPeptide', 'glycosylation', 'propeptide',
                             'disulfideBinary', 'intMetBinary', 'intramembraneBinary',
                             'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary',
                             'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
                             'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
                             'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                             'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary',
                             'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
                             'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
                             'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
                             'glycosylationBinary', 'propeptideBinary', 'from', 'to', 'template',
                             'chain', 'score', 'pdbID', 'pdbSequence', 'pdb_alignStatus',
                             'mutationPositionOnPDB', 'domainStartonPDB', 'domainEndonPDB'])
                modbase_match = which_ones_are_match[which_ones_are_match.mutationPositionOnPDB != 'nan']
                modbase_not_match = which_ones_are_match[which_ones_are_match.mutationPositionOnPDB == 'nan']

            else:
                modbase_match = which_ones_are_match[which_ones_are_match.mutationPositionOnPDB != 'nan']
                modbase_not_match = which_ones_are_match[which_ones_are_match.mutationPositionOnPDB == 'nan']

            which_ones_are_match = None
            modbase_match.score = modbase_match.score.astype('float')
            modbase_match = modbase_match.sort_values(['datapoint', 'mutationPositionOnPDB', 'score'],
                                                      ascending=[True, True, False])
            modbase_match.drop_duplicates(['datapoint'], keep='first', inplace=True)
            not_nan = None
            nan = None


            # merge not_in_align and modbase_not_match as they were both excluded from modbase match.

            # No model
            no_info = no_info[to_swiss_columns]
            no_info = no_info.drop_duplicates()

            # Model present, no sequence
            not_models = not_models[to_swiss_columns]
            not_models = not_models.drop_duplicates()

            # Modbase model and sequence present, no match in PDB
            modbase_not_match = modbase_not_match[to_swiss_columns]
            modbase_not_match = modbase_not_match.drop_duplicates()
            if len(not_in_aligned) != 0 and len(modbase_not_match) != 0 and len(no_info) != 0:
                rest = pd.concat([not_in_aligned, modbase_not_match, no_info])
            elif len(not_in_aligned) != 0 and len(modbase_not_match) != 0 and len(no_info) == 0:
                rest = pd.concat([not_in_aligned, modbase_not_match])
            elif len(not_in_aligned) == 0 and len(modbase_not_match) != 0 and len(no_info) != 0:
                rest = pd.concat([modbase_not_match, no_info])
            elif len(not_in_aligned) != 0 and len(modbase_not_match) == 0 and len(no_info) != 0:
                rest = pd.concat([not_in_aligned, no_info])
            elif len(not_in_aligned) != 0 and len(modbase_not_match) == 0 and len(no_info) == 0:
                rest = not_in_aligned
            elif len(not_in_aligned) == 0 and len(modbase_not_match) != 0 and len(no_info) == 0:
                rest = modbase_not_match
            elif len(not_in_aligned) == 0 and len(modbase_not_match) == 0 and len(no_info) != 0:
                rest = no_info
            else:
                rest = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                                             'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                             'wt_sequence_match', 'whichIsoform', 'datapoint'])

            rest = rest[to_swiss_columns]
            rest = rest.drop_duplicates()

            rest.reset_index(inplace=True)
            rest.drop(['index'], axis=1, inplace=True)
            rest = rest.astype('str')


        else:

            modbase_match = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                                                  'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                                  'wt_sequence_match', 'whichIsoform', 'datapoint', 'disulfide', 'intMet',
                                                  'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                                                  'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink',
                                                  'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                                                  'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                                                  'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil',
                                                  'peptide', 'transitPeptide', 'glycosylation', 'propeptide',
                                                  'disulfideBinary', 'intMetBinary', 'intramembraneBinary',
                                                  'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary',
                                                  'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
                                                  'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
                                                  'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                                                  'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary',
                                                  'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
                                                  'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
                                                  'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
                                                  'glycosylationBinary', 'propeptideBinary', 'from', 'to', 'template',
                                                  'chain', 'score', 'pdbID', 'pdbSequence', 'pdb_alignStatus',
                                                  'mutationPositionOnPDB', 'domainStartonPDB', 'domainEndonPDB'])
            not_in_aligned = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume', 'granthamScore',
                                                   'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                                   'wt_sequence_match', 'whichIsoform', 'datapoint', 'disulfide', 'intMet',
                                                   'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                                                   'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink',
                                                   'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                                                   'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                                                   'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil',
                                                   'peptide', 'transitPeptide', 'glycosylation', 'propeptide', 'disulfide',
                                                   'intMet', 'intramembrane', 'naturalVariant', 'dnaBinding', 'activeSite',
                                                   'nucleotideBinding', 'lipidation', 'site', 'transmembrane', 'crosslink',
                                                   'mutagenesis', 'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                                                   'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                                                   'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil',
                                                   'peptide', 'transitPeptide', 'glycosylation', 'propeptide', 'from',
                                                   'to', 'template', 'chain', 'score', 'pdbID', 'pdbSequence', 'fasta'])
            no_info = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                                            'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                            'wt_sequence_match', 'whichIsoform', 'datapoint'])
            rest = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume', 'granthamScore',
                                         'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                         'wt_sequence_match', 'whichIsoform', 'datapoint'])

            rest = rest[to_swiss_columns]
            rest = rest.drop_duplicates()

            rest.reset_index(inplace=True)
            rest.drop(['index'], axis=1, inplace=True)
            rest = rest.astype('str')
            to_modbase_size = 0

        print('Modbase matching is completed...\n')
        print('SUMMARY')
        print('-------')
        print('%d data points that failed to match a UniProt Sequence are discarded.' % len(
            not_match_in_uniprot.drop_duplicates(['datapoint'])))
        print('Of the remaining %d:' % uniprot_matched_size)
        print('--%d of %d successfully aligned with PDB structures.' % (
            len(pdb_aligned.drop_duplicates(['datapoint'])), with_pdb_size))
        print('--%d of %d successfully aligned with SwissModels structures.' % (
            len(swiss_match.drop_duplicates(['datapoint'])), to_swiss_size))
        print('--%d of %d successfully aligned with Modbase structures.\n' % (
            len(modbase_match.drop_duplicates(['datapoint'])), to_modbase_size))
        print('--Remaining %d not found to match any models.' % len(rest.drop_duplicates(['datapoint'])))
        print('--A total of %d datapoints will not be evaluated.\n' % (
                len(rest.drop_duplicates(['datapoint'])) + len(not_match_in_uniprot.drop_duplicates(['datapoint']))))

        print('FOR CHECKING : ',
              len(rest.drop_duplicates(['datapoint'])) + len(not_match_in_uniprot.drop_duplicates(['datapoint'])) + len(
                  pdb_aligned.drop_duplicates(['datapoint'])) + len(swiss_match.drop_duplicates(['datapoint'])) + len(
                  modbase_match.drop_duplicates(['datapoint'])) == data_size)
        no_info = None
        align = None
        not_in_aligned = None
        not_models = None
        modbase_not_match = None


        # Final corrections

        # Now 3D alignment.
        pdb = pdb_aligned.copy()
        swiss = swiss_match.copy()
        modbase = modbase_match.copy()
        pdb_aligned = None
        swiss_match = None
        modbase_match = None

        """
        WHAT DO WE HAVE NOW?
        - uniprot sequence not found
        - pdb aligned
        - swiss aligned
        - modbase aligned
        - not aligned with anything (rest)
        """

        # Fix the axes and  merge all data.


        pdb.drop(['pdbInfo'], axis=1, inplace=True)
        pdb.rename(columns={'resolution': 'score'}, inplace=True)
        swiss.rename(columns={'qmean_norm': 'score'}, inplace=True)
        modbase.rename(columns={'qmean_norm': 'score'}, inplace=True)

        swiss = swiss[pdb.columns]
        modbase = modbase[pdb.columns]
        pdb['source'] = 'PDB'
        swiss['source'] = 'SWISSMODEL'
        modbase['source'] = 'MODBASE'
        data = pd.concat([swiss, modbase, pdb])


        data.reset_index(inplace=True)
        data.drop(['index'], axis=1, inplace=True)
        data = data.astype('str')
        data_spare = pd.concat([not_match_in_uniprot, rest])
        not_match_in_uniprot = None
        pdb = None
        swiss = None
        modbase = None
        rest = None

        print('Generating FreeSASA files...')
        print('------------------------------------\n')
        # Folder to calculated RSA values.

        existing_free_sasa = list(Path(path_to_output_files / 'freesasa_files').glob("*"))
        existing_free_sasa = [str(i) for i in existing_free_sasa]
        existing_free_sasa = [i.split('/')[-1].split('.')[0] for i in existing_free_sasa]

        print('Calculation RSA for PDB Structure Files...\n')

        pdb_only = data[data.source == 'PDB']
        for pdbID in pdb_only.pdbID.to_list():
            if pdbID not in existing_free_sasa:
                (run_freesasa(Path(path_to_output_files / 'pdb_structures' / f'{pdbID.lower()}.pdb'),
                              Path(path_to_output_files / 'freesasa_files' / f'{pdbID.lower()}.txt'), include_hetatms=True,
                              outdir=None, force_rerun=False, file_type='pdb'))


        print('Calculation RSA for SwissModel Files...\n')
        swiss_only = data[data.source == 'SWISSMODEL']
        swiss_dp = []
        for i in swiss_only.index:
            swiss_dp.append(swiss_only.at[i, 'uniprotID'] + '_' + swiss_only.at[i, 'pdbID'].lower() + '_' + str(
                round(float(swiss_only.at[i, 'score']), 2)))
        for pdbID in swiss_dp:
            if pdbID not in existing_free_sasa:
                (run_freesasa(Path(path_to_output_files / 'swissmodel_structures' / f'{pdbID}.txt'),
                              Path(path_to_output_files / 'freesasa_files' / f'{pdbID}.txt'), include_hetatms=True,
                              outdir=None, force_rerun=False, file_type='pdb'))

        print('Calculation RSA for Modbase Model Files...\n')
        modbase_only = data[data.source == 'MODBASE']
        for pdbID in modbase_only.pdbID.to_list():
            if pdbID not in existing_free_sasa:
                (run_freesasa(Path(path_to_output_files / 'modbase_structures_individual' / f'{pdbID.lower()}.txt'),
                              Path(path_to_output_files / 'freesasa_files' / f'{pdbID.lower()}.txt'), include_hetatms=True,
                              outdir=None, force_rerun=False, file_type='pdb'))

        # This annotation list is different than the prev one, keep it.

        annotation_list += ['domainStartonPDB', 'domainEndonPDB']

        folder_path = path_to_output_files / 'freesasa_files'

        aligner = Align.PairwiseAligner()
        print('Proceeding to 3D distance calculation...\n')

        data.domainEndonPDB = data.domainEndonPDB.astype(str)
        data.domainStartonPDB = data.domainStartonPDB.astype(str)

        existing_free_sasa = None
        swiss_dp = None
        pdb_only = None
        swiss_only = None
        modbase_only = None
        data['uniprotSequence'] = data['uniprotSequence'].str.replace('U', 'C')
        data['pdbSequence'] = data['pdbSequence'].str.replace('U', 'C')
        for i in data.index:
            id_ = data.at[i, 'pdbID'].lower()
            up_id_ = data.at[i, 'uniprotID']
            score_ = str(data.at[i, 'score'])
            if data.at[i, 'source'] == 'PDB':
                pdb_path = Path(path_to_output_files / 'pdb_structures' / f'{id_}.pdb')
            elif data.at[i, 'source'] == 'MODBASE':
                pdb_path = Path(path_to_output_files / 'modbase_structures_individual' / f'{id_}.txt')
            elif data.at[i, 'source'] == 'SWISSMODEL':
                pdb_path = Path(path_to_output_files / 'swissmodel_structures' / f'{up_id_}_{id_}_{score_}.txt')

            pdbSequence = data.at[i, 'pdbSequence']
            source = data.at[i, 'source']
            chain = data.at[i, 'chain']
            uniprotID = data.at[i, 'uniprotID']
            pdbID = data.at[i, 'pdbID']
            alignments = get_alignments_3D(uniprotID, 'nan', pdb_path, pdbSequence, source, chain, pdbID, mode, Path(path_to_output_files / '3D_alignment'), file_format = 'gzip')
            mutPos = data.at[i, 'mutationPositionOnPDB']
            try:
                coordMut = get_coords(mutPos, alignments , 'nan', 'nan', mode)[0]
            except:
                ValueError
                coordMut = 'nan'
            try:
                sasa_pos = get_coords(mutPos, alignments, 'nan', 'nan', mode)[2]
                data.at[i, 'sasa'] = sasa(data.at[i, 'source'], data.at[i, 'pdbID'], data.at[i, 'uniprotID'], sasa_pos, data.at[i, 'wt'], mode, path_to_output_files,file_type = 'pdb')
            except:
                ValueError
                data.at[i, 'sasa'] = 'nan'  # mutation position is nan
            for annot in annotation_list:
                annotx = []
                try:
                    positions_of_annotations = data.at[i, annot].split(',')
                    for pos in positions_of_annotations:
                        pos = pos.strip().strip('\'').strip('[\'').strip('\']')
                        try:
                            if '-' not in pos:
                                pos = int(float(pos))
                                coordAnnot = get_coords(pos, alignments, 'nan', 'nan', mode)[0]
                                try:
                                    annotx.append(find_distance(coordMut, coordAnnot))
                                except:
                                    ValueError

                            else:
                                for r in range(int(pos.split('-')[0]), int(pos.split('-')[1]) + 1):
                                    coordAnnot = get_coords(r, alignments, 'nan', 'nan', mode)[0]
                                    annotx.append(find_distance(coordMut, coordAnnot))
                        except:
                            ValueError
                    try:
                        data.at[i, annot] = min([float(i) for i in annotx])
                    except:
                        ValueError
                        data.at[i, annot] = 'nan'

                except:
                    ValueError

            if (str(data.at[i, 'domainStartonPDB']) == 'NaN' or str(data.at[i, 'domainStartonPDB']) == 'nan') and (
                    str(data.at[i, 'domainEndonPDB']) != 'NaN' and str(data.at[i, 'domainEndonPDB']) != 'nan'):
                data.at[i, 'domainStartonPDB'] = 100000
            elif (str(data.at[i, 'domainEndonPDB']) == 'NaN' or str(data.at[i, 'domainEndonPDB']) == 'nan') and (
                    str(data.at[i, 'domainStartonPDB']) != 'NaN' and str(data.at[i, 'domainStartonPDB']) != 'nan'):
                data.at[i, 'domainEndonPDB'] = 100000
            elif (str(data.at[i, 'domainStartonPDB']) == 'NaN' and str(data.at[i, 'domainEndonPDB']) == 'nan'):
                data.at[i, 'domaindistance3D'] = 'nan'

            data.at[i, 'domaindistance3D'] = min(float(data.at[i, 'domainStartonPDB']),
                                                 float(data.at[i, 'domainEndonPDB']))
            data.at[i, 'domaindistance3D'] = min(float(data.at[i, 'domainStartonPDB']),
                                                 float(data.at[i, 'domainEndonPDB']))


        data = data.astype(str)
        data.replace({'NaN': 'nan'}, inplace=True)


        # Now unify all 3 separate data. We have with_pdb. The ones that have pdb structyres, swiss, modbase, the ones didnt match with ant and the ones didnt have wt seq match.

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

        if len(data) == 0:
            data = pd.DataFrame(columns=['uniprotID', 'wt', 'mut', 'pos', 'composition', 'polarity', 'volume','granthamScore',
                                         'domain', 'domStart', 'domEnd', 'distance', 'uniprotSequence',
                                         'pdbSequence', 'wt_sequence_match', 'whichIsoform', 'pdbID', 'score',
                                         'chain', 'datapoint', 'disulfide', 'intMet', 'intramembrane',
                                         'naturalVariant', 'dnaBinding', 'activeSite', 'nucleotideBinding',
                                         'lipidation', 'site', 'transmembrane', 'crosslink', 'mutagenesis',
                                         'strand', 'helix', 'turn', 'metalBinding', 'repeat',
                                         'topologicalDomain', 'caBinding', 'bindingSite', 'region',
                                         'signalPeptide', 'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil',
                                         'peptide', 'transitPeptide', 'glycosylation', 'propeptide',
                                         'disulfideBinary', 'intMetBinary', 'intramembraneBinary',
                                         'naturalVariantBinary', 'dnaBindingBinary', 'activeSiteBinary',
                                         'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
                                         'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
                                         'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                                         'repeatBinary', 'topologicalDomainBinary', 'caBindingBinary',
                                         'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
                                         'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
                                         'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
                                         'glycosylationBinary', 'propeptideBinary', 'pdb_alignStatus',
                                         'mutationPositionOnPDB', 'domainStartonPDB', 'domainEndonPDB',
                                         'source', 'sasa', 'domaindistance3D', 'threeState_trsh4_HQ', 'domain_fisher'])
        else:
            data.sasa = data.sasa.astype('str')

        for i in data.index:
            if '*' in data.at[i, 'sasa']:
                data.at[i, 'sasa'] = data.at[i, 'sasa'].split('*')[0]

        data.sasa = data.sasa.replace({'N/A': 'nan'})
        data.sasa = data.sasa.replace({'None': 'nan'})
        data.replace({'   N/A': 'nan'}, inplace=True)
        data.replace({'None': 'nan'}, inplace=True)
        data.sasa = data.sasa.astype(float)
        data = data.astype(str)
        for i in data.index:
            if float(data.at[i, 'sasa']) < 5:
                data.at[i, 'trsh4'] = 'core'
            elif float(data.at[i, 'sasa']) >= 5:
                data.at[i, 'trsh4'] = 'surface'
            elif data.at[i, 'sasa'] == 'nan':
                data.at[i, 'trsh4'] = 'nan'

        data = data.merge(interface_dataframe, on='uniprotID', how='left')
        data.positions = data.positions.astype('str')
        for i in data.index:
            if (str(data.at[i, 'pos']) in data.at[i, 'positions']) and data.at[i, 'trsh4'] == 'surface':
                data.at[i, 'threeState_trsh4_HQ'] = 'interface'
            elif (str(data.at[i, 'pos']) not in data.at[i, 'positions']) and data.at[i, 'trsh4'] == 'surface':
                data.at[i, 'threeState_trsh4_HQ'] = 'surface'
            elif (str(data.at[i, 'pos']) not in data.at[i, 'positions']) and data.at[i, 'trsh4'] == 'core':
                data.at[i, 'threeState_trsh4_HQ'] = 'core'
            elif (str(data.at[i, 'pos']) in data.at[i, 'positions']) and data.at[i, 'trsh4'] == 'core':
                data.at[i, 'threeState_trsh4_HQ'] = 'conflict'
            elif data.at[i, 'trsh4'] == 'nan':
                data.at[i, 'threeState_trsh4_HQ'] = 'nan'

        data.drop(['positions'], axis=1, inplace=True)


        # OPTIONAL
        # DOMAIN SELECTION
        # Next step: Delete all other domains with 'NULL.' R is capable of handling 53 categories. We will keep 52 most
        # significant domains and 53th category will be NULL.

        fisherResult = pd.read_csv(fisher_path, sep='\t')

        significant_domains = fisherResult.domain.to_list()
        for i in data.index:
            if data.at[i, 'domain'] in significant_domains:
                data.at[i, 'domain_fisher'] = data.at[i, 'domain']
            else:
                data.at[i, 'domain_fisher'] = 'NULL'

        # Change the numbering for binary annotations and create 3 classes:
        # nan--> 0, 0 -->1 and 1 -->2

        print('Final adjustments are being done...\n')
        binaryCols = ['disulfideBinary', 'intMetBinary', 'intramembraneBinary', 'naturalVariantBinary', 'dnaBindingBinary',
                      'activeSiteBinary', 'nucleotideBindingBinary', 'lipidationBinary', 'siteBinary',
                      'transmembraneBinary', 'crosslinkBinary', 'mutagenesisBinary',
                      'strandBinary', 'helixBinary', 'turnBinary', 'metalBindingBinary',
                      'repeatBinary', 'caBindingBinary', 'topologicalDomainBinary',
                      'bindingSiteBinary', 'regionBinary', 'signalPeptideBinary',
                      'modifiedResidueBinary', 'zincFingerBinary', 'motifBinary',
                      'coiledCoilBinary', 'peptideBinary', 'transitPeptideBinary',
                      'glycosylationBinary', 'propeptideBinary']
        data = data.astype(str)
        data.replace({'NaN': 'nan'}, inplace=True)
        for i in data.index:
            for j in binaryCols:
                data[j] = data[j].astype('str')
                if (data.at[i, j] == '0') or (data.at[i, j] == '0.0'):
                    data.at[i, j] = '1'
                elif data.at[i, j] == 'nan':
                    data.at[i, j] = '0'
                elif (data.at[i, j] == '1') or (data.at[i, j] == '1.0'):
                    data.at[i, j] = '2'

        annotCols = ['disulfide', 'intMet', 'intramembrane',
                     'naturalVariant', 'dnaBinding', 'activeSite', 'nucleotideBinding',
                     'lipidation', 'site', 'transmembrane', 'crosslink', 'mutagenesis',
                     'strand', 'helix', 'turn', 'metalBinding', 'repeat', 'caBinding',
                     'topologicalDomain', 'bindingSite', 'region', 'signalPeptide',
                     'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil', 'peptide',
                     'transitPeptide', 'glycosylation', 'propeptide']

        for i in data.index:
            for annot in annotCols:
                binaryName = str(annot) + 'Binary'
                if data.at[i, binaryName] == '2':
                    data.at[i, annot] = '0.0'
        data.replace({'100000': 'nan'}, inplace=True)
        data = add_physicochemical(data)
        data.rename(
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

        data = data[
            ['prot_uniprotAcc', 'wt_residue', 'mut_residue', 'position', 'meta_merged', 'composition', 'polarity',
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
        ready = data.copy()
        # Imputation
        if (impute == 'True') or (impute == 'true'):
            filler = [17.84, 30.8, 24.96, 13.12, 23.62, 18.97, 20.87, 29.59, 20.7, 12.7, 22.85, 17.21, 9.8, 9, 15.99, 16.82,
                      20.46, 24.58, 9.99, 17.43, 20.08, 30.91, 20.86, 22.14, 21.91, 28.45, 17.81, 25.12, 20.33, 22.36]
            col_index = 0
            for col_ in ready.columns[-30:]:
                ready[col_] = ready[col_].fillna(filler[col_index])
                ready[col_] = ready[col_].replace({'nan': filler[col_index]})
                col_index += 1
            ready['domains_3Ddist'] = ready['domains_3Ddist'].fillna(24.5)
            ready['sasa'] = ready['sasa'].fillna(29.5)
            ready['location_3state'] = ready['location_3state'].fillna('unknown')
        elif (impute == 'False') or (impute == 'false'):
            pass
        ready = ready.replace({'nan': np.NaN})
        ready.to_csv(path_to_output_files / 'featurevector_pdb.txt', sep='\t', index=False)
        if len(ready) == 0:
            print('No feature vector could be produced for input data. Please check the presence of a structure for the input proteins.')
        print(ready)
        print('Feature vector successfully created...')
        return ready

    end = timer()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    sys.stdout.close()
    return ready
