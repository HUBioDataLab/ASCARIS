import ast
import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
import unipressed
from requests.adapters import HTTPAdapter, Retry
from unipressed import IdMappingClient
import Bio
from Bio import SeqIO
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import *
from io import StringIO
from utils import *

import math

import json
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
def get_pdb_ids(protein_id):
    try:
        request = IdMappingClient.submit(
            source="UniProtKB_AC-ID", dest="PDB", ids={protein_id})
        pdb_list = list(request.each_result())
        return [i['to'] for i in pdb_list]
    except requests.exceptions.HTTPError:
        return  []
    except unipressed.id_mapping.core.IdMappingError:
        print('IdMappingError caused by UniProt API service, please try later.')
        return  []
    except KeyError:
        return  []


def fix_filename(filename):
    try:
        if Path(filename).suffix == '.pdb':
            pass
        elif Path(filename).stem.endswith("ent"):
            filename_replace_ext = filename.with_name( Path(filename).stem[3:])
            Path(filename).rename(filename_replace_ext.with_suffix('.pdb'))
        elif Path(filename).stem.startswith("pdb"):
            filename_replace_ext = Path(filename).with_name(Path(filename).stem[3:])
            Path(filename).rename(filename_replace_ext.with_suffix('.pdb'))
        else:
            filename_replace_ext = filename.with_suffix(".pdb")
            Path(filename).rename(filename_replace_ext)

    except:
        FileNotFoundError



def fetch_uniprot_ids(pdb_code):
    response = requests.get(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}")
    response.raise_for_status()
    resp = response.json()
    return list(list(list(resp.values())[0].values())[0].keys())

def addPDBinfo(data, path_to_output_files):
    # pdb_fasta = pd.DataFrame(columns=['pdbID', 'chain', 'pdbSequence'])
    pdb_info = pd.DataFrame(columns=['uniprotID', 'pdbID', 'chain', 'resolution'])
    print('Retrieving PDB structures...\n')
    up_list = data.uniprotID.to_list()
    pdbs = [get_pdb_ids(i) for i in up_list]

    if len(pdbs) >= 1:
        pdbs = [item for sublist in pdbs for item in sublist]
        pdbs = list(filter(None, pdbs))
        pdbs = set(pdbs)
        pdbs = [i.lower() for i in pdbs]
    else:
        pdbs = []
        print('No PDB structure found for the query. ')

    print('\n>>Starting PDB structures download...\n')
    print('\n>>Processing PDB structures...\n')
    parser = PDBParser()
    ppb = PPBuilder()

    index = 0
    for search in pdbs:
        print(f'Searching for {search.upper()}')
        try:
            pdb_url = f"https://files.rcsb.org/download/{search}.pdb"
            response = requests.get(pdb_url)
            response.raise_for_status()  # Check for a successful response
            pdb_data = response.text
            pdb_parser = PDBParser(QUIET=True)  # QUIET=True suppresses warnings
            pdb_file_content = StringIO(pdb_data)
            structure = pdb_parser.get_structure(search, pdb_file_content)
            pdb_data_list = pdb_data.split('\n')
            pdb_data_list = [i for i in pdb_data_list if i.startswith('DBREF')]
            pdb_data_list = [[list(filter(None, i.split(' '))) for j in i.split(' ') if j == 'UNP'] for
                             i in pdb_data_list]
            pdb_data_list = [i for i in pdb_data_list if i != []]
            header = structure.header
            for unp in pdb_data_list:
                if (unp[0][5] == 'UNP') & (unp[0][6].split('-')[0] in up_list):
                    pdb_info.at[index, 'uniprotID'] = unp[0][6].split('-')[0]
                    pdb_info.at[index, 'pdbID'] = unp[0][1].upper()
                    pdb_info.at[index, 'chain'] = unp[0][2].upper()
                    pdb_info.at[index, 'resolution'] = header.get('resolution', 'N/A')
                    pdb_info.at[index, 'start'] = unp[0][8]
                    pdb_info.at[index, 'end'] = unp[0][9]
                    index += 1
        except:
            continue
    pdb_info.replace({'None': np.NaN}, inplace=True)
    print('PDB file processing finished..')

    return pdb_info
from add_sasa import *



def downloadPDB(pdbID, path_to_output_files):
    pdbl = PDBList()
    existing_pdb = list(Path(path_to_output_files / 'pdb_structures').glob("*"))
    existing_pdb = [str(i) for i in existing_pdb]
    existing_pdb = [i.split('/')[-1].split('.')[0].lower() for i in existing_pdb]
    if pdbID not in existing_pdb:
        # print(f'Downloading PDB file for {pdbID.upper()}..')
        file = pdbl.retrieve_pdb_file(pdbID, pdir=Path(path_to_output_files / 'pdb_structures'), file_format="pdb")
        fix_filename(file)
        file = fix_filename(file)
        file = Path(path_to_output_files / 'pdb_structures' / f'{pdbID}.pdb')
    else:
        print(f'PDB file for {pdbID.upper()} exists..')
        file = Path(path_to_output_files / 'pdb_structures' / f'{pdbID}.pdb')
        fix_filename(file)
        file = fix_filename(file)

        file = Path(path_to_output_files / 'pdb_structures' / f'{pdbID}.pdb')


    existing_free_sasa = list(Path(path_to_output_files / 'freesasa_files').glob("*"))
    existing_free_sasa = [str(i) for i in existing_free_sasa]
    existing_free_sasa = [i.split('/')[-1].split('.')[0] for i in existing_free_sasa]
    if pdbID not in existing_free_sasa:
        run_freesasa(file, Path(path_to_output_files / 'freesasa_files' / f'{pdbID}.txt'), include_hetatms=True,
                              outdir=None, force_rerun=False, file_type='pdb')

    return file

def processFile(data, path_to_output_files):
    for i in data.index:
        protein = data.at[i,'uniprotID']
        pdbID = data.at[i,'pdbID'].lower()
        chain =  data.at[i,'chain']
        pos = int(data.at[i, 'pos'])
        wt = data.at[i, 'wt']


        url = f'https://files.rcsb.org/download/{pdbID}.pdb'
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(f'{path_to_output_files}/pdb_structures/{pdbID}.pdb', 'w') as f:
                f.write(response.text)
            print(f"Downloaded {pdbID}.pdb successfully.")
        else:
            print(f"Failed to download {pdbID}.pdb. Status code: {response.status_code}")
        file = Path(path_to_output_files / 'pdb_structures' / f'{pdbID}.pdb')

        
        run_freesasa(file, Path(path_to_output_files / 'freesasa_files' / f'{pdbID}.txt'), include_hetatms=True,
                                  outdir=None, force_rerun=False, file_type='pdb')

 
        
        filename = Path(path_to_output_files / 'freesasa_files' / f'{pdbID}.txt')
        data.loc[i, 'sasa'] = sasa(protein, pos, wt, 1, filename, path_to_output_files,file_type='pdb')

        
        
        newCol = {}
        with open(file, encoding="utf8") as f:
            for line in f.readlines():
                if line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA' and line[21].upper() == chain.upper():
                    coords= [line[31:38].strip(), line[39:46].strip(), line[47:54].strip()]
                    resnums_for_sasa = line[22:26].strip()
                    newCol[resnums_for_sasa] = coords
                elif line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA' and line[21] == ' ':
                    coords= [line[31:38].strip(), line[39:46].strip(), line[47:54].strip()]
                    resnums_for_sasa = line[22:26].strip()
                    newCol[resnums_for_sasa] = coords
        data.at[i, 'coordinates'] = json.dumps(newCol)
    return data

def distance(x1, y1, z1, x2, y2, z2):
    d = math.sqrt(math.pow(x2 - x1, 2) +
                  math.pow(y2 - y1, 2) +
                  math.pow(z2 - z1, 2) * 1.0)
    return d


def find_distance(coordMut, coordAnnot):
    if coordMut != np.NaN:
        try:
            dist = distance(float(coordMut[0]), float(coordMut[1]), float(coordMut[2]), float(coordAnnot[0]),
                            float(coordAnnot[1]), float(coordAnnot[2]))

            return "%.2f" % dist
        except:
            ValueError
            dist = 'nan'
            return dist
    else:
        return np.NaN

def domainDistance(domStart, domEnd, coordinates, mutationPosition, matchList, posOnPDB):
    resList = list(range(domStart, domEnd))
    domainDistanceList = []
    for i in resList:
        try:
            domainPos = ast.literal_eval(matchList)[str(i)]
            coordMut = coordinates[str(posOnPDB)]
            coordDomain = coordinates[str(domainPos)]
            distance = find_distance(coordMut, coordDomain)
            domainDistanceList.append(distance)
            return min(domainDistanceList)
        except KeyError:
            domainDistanceList = np.NaN
            return np.NaN



def match3D(data):
    data.fillna(np.NaN, inplace=True)
    for i in data.index:
        coordinates = ast.literal_eval(data.at[i, 'coordinates'])
        pos = str(data.at[i, 'pos'])
        matchList = data.at[i, 'MATCHDICT']
        try:
            posOnPDB = ast.literal_eval(data.at[i, 'MATCHDICT'])[pos]
            coordMut = coordinates[str(posOnPDB)]
            if data.at[i, 'distance'] == -1000:
                domStart = data.at[i, 'domStart']
                domEnd = data.at[i, 'domEnd']
                data.at[i, 'distance'] = domainDistance(domStart, domEnd, coordinates, pos, matchList, posOnPDB)
        except KeyError:
            posOnPDB = np.NaN
            coordMut = np.NaN
            data.at[i, 'distance'] = np.NaN


        for col in UNIPROT_ANNOTATION_COLS[0:30]:
            allDist = []
            if (data.at[i, col] != np.NaN) & (data.at[i, col] != 'hit') &  (data.at[i, col] != '[]')&  (data.at[i, col] != []):
                annotation_list = ast.literal_eval(data.at[i, col])
                integer_list = [int(element) for element in annotation_list if element != 'null']
                for annotPosition in integer_list:
                    coordAnnot = coordinates[str(annotPosition)]
                    distance = find_distance(coordMut, coordAnnot)
                    allDist.append(distance)
                if len(allDist)>0:
                    data.at[i, col] = min(allDist)
    return data


def domainDistanceModels(domStart, domEnd, coordinates, mutationPosition):
    resList = list(range(domStart, domEnd))
    domainDistanceList = []
    for i in resList:
        try:
            coordMut = (coordinates)[mutationPosition]
            coordDomain = (coordinates)[i]
            distance = find_distance(coordMut, coordDomain)
            domainDistanceList.append(distance)
            return min(domainDistanceList)
        except KeyError:
            domainDistanceList = np.NaN
            return np.NaN


def match3DModels(data):
    data.fillna(np.NaN, inplace=True)
    for i in data.index:
        pos = int(data.at[i, 'pos'])
        coords = data.at[i, 'coordinates']
        if type(coords) != dict:
            coordinates = ast.literal_eval(coords)
        else:
            coordinates = coords
            pass
        coordMut = coordinates[pos]
        if data.at[i, 'distance'] == -1000:
            domStart = data.at[i, 'domStart']
            domEnd = data.at[i, 'domEnd']
            data.at[i, 'distance'] = domainDistanceModels(domStart, domEnd, coordinates, pos)
        for col in UNIPROT_ANNOTATION_COLS[0:30]:
            allDist = []
            if (data.at[i, col] != np.NaN) & (data.at[i, col] != 'hit') &  (data.at[i, col] != '[]')&  (data.at[i, col] != []):
                annotation_list = ast.literal_eval(data.at[i, col])
                integer_list = [int(element) for element in annotation_list]
                for annotPosition in integer_list:
                    try:
                        coordAnnot = coordinates[annotPosition]
                    except KeyError:
                        coordAnnot = []
                    distance = find_distance(coordMut, coordAnnot)
                    allDist.append(distance)

                if len(allDist)>0:
                    allDist = [float(i) for i in allDist]
                    data.at[i, col] = min(allDist)

    return data


def selectMaxAnnot(data):
    if len(data) >0:
        for i in data.index:
            total = 0
            nanCounter = 0
            hitCounter = 0
            for col in UNIPROT_ANNOTATION_COLS[0:30]:
                if (str(data.at[i,col]) != 'nan') and (data.at[i,col] != '[]') and (data.at[i,col] != 'hit') and (data.at[i,col] != ''):
                    total += float(data.at[i,col])
                elif  (str(data.at[i,col]) == 'nan') or (data.at[i,col] == '[]') or (data.at[i,col] != ''):
                    nanCounter +=1
                if data.at[i,col] == 'hit':
                    hitCounter += 1

            if hitCounter > 0:
                data.at[i, 'hitTotal'] = hitCounter
            else:
                data.at[i, 'hitTotal'] = np.NaN

            if nanCounter != 30:
                data.at[i, 'annotTotal'] = total
            else:
                data.at[i, 'annotTotal'] = np.NaN
    else:
        data['annotTotal'] = np.NaN

    return data
