import numpy as np
import pandas as pd
import os
import requests
import json
import tarfile, gzip
import time, glob
from utils import threeToOne
import streamlit as st
from pathlib import Path
import gzip
import shutil
import codecs
import io

def uniprot_pdb_residue_mapping(pdb_id, uniprot_id, save_path):

  """
  This code does residue-wise mapping between UniProt and PDB residues.
  """
  ascaris = {}
  full_ascaris = {}
    
  res = requests.get(f'https://www.ebi.ac.uk/pdbe/download/api/pdb/entry/sifts?id={pdb_id}')
  url = json.loads(res.text)['url']
  response = requests.get(url, stream=True)

  file = tarfile.open(fileobj=response.raw, mode="r|gz")
  file.extractall(path=save_path)  # Creates another gz file


  existing_pdb = list(Path(save_path).glob("*"))
  existing_pdb = [str(i) for i in existing_pdb]

  try:
      with gzip.open(f'{save_path}/{pdb_id.lower()}.xml.gz', 'rt') as f:
          file_content = f.read()
  except FileNotFoundError:
      with gzip.open(f'{save_path}/{pdb_id}.xml.gz', 'rt') as f:
          file_content = f.read()
  content = file_content.split('\n')
  index = [idx for idx, s in enumerate(content) if 'listResidue' in s]
  listResidues = []
  for ind in range(0, len(index), 2):
      try:
          if ((content[index[ind]]).strip() == '<listResidue>') & (
                  (content[index[ind + 1]]).strip() == '</listResidue>'):
              listResidues.append(content[index[ind]:index[ind + 1]])
      except:
          IndexError
  for true_content in listResidues:
      for sub_content in true_content:
          if f'dbAccessionId="{uniprot_id}"' in sub_content:
              content = [i.strip() for i in true_content]
              sel = [i for i in content if
                     ('<crossRefDb dbSource="PDB"' in i or '<crossRefDb dbSource="UniProt"' in i)]
              matching_dict = {}
              if len(sel) % 2 == 0:  # if correct residues
                  dbAccessionId = [i.split('dbAccessionId')[1].split(' ')[0].split('=')[1].strip('"').upper() for i
                                   in sel]
                  dbSource = [i.split('dbSource')[1].split(' ')[0].split('=')[1].strip('"').upper() for i in sel]
                  dbResNum = [i.split('dbResNum')[1].split(' ')[0].split('=')[1].strip('"') for i in sel]
                  dbResName = [i.split('dbResName')[1].split(' ')[0].split('=')[1].split('/')[0].strip('"') for i in
                               sel]
                  dbChainName = [i.split('dbChainId')[1].split(' ')[0].split('=')[1].split('/')[0].strip('"') for i
                                 in sel if 'crossRefDb dbSource="PDB' in i]

                  for k, j, m in zip(range(0, len(dbAccessionId), 2), range(1, len(dbAccessionId) - 1, 2), range(len(dbChainName))):
                      # try:
                      if dbResName[j] == threeToOne(dbResName[k]) and dbAccessionId[j] == uniprot_id:
                          matching_dict[
                              dbSource[j] + '_' + dbAccessionId[j] + '_' + dbResNum[j] + '_' + dbResName[j]] = \
                              dbSource[k] + '_' + dbAccessionId[k] + '_' + dbResNum[k] + '_' + threeToOne(
                                  dbResName[k]) + '_' + dbChainName[m]
                      # except:
                      #     KeyError

              only_residues = {k.split('_')[2]: v.split('_')[2] for k, v in matching_dict.items()}
              for k, v in matching_dict.items():
                  if v.split('_')[1] + v.split('_')[-1] not in ascaris.keys():
                      ascaris[v.split('_')[1] + v.split('_')[-1]] = only_residues
              for k, v in matching_dict.items():
                  if v.split('_')[1] + v.split('_')[-1] not in full_ascaris.keys():
                      full_ascaris[v.split('_')[1] + v.split('_')[-1]] = matching_dict

  return ascaris ,full_ascaris

import ast
def pdbMapping(data, save_path): # BU DATA hangi df hepi mi azalttigimiz mi
    # Here we add match dictionary containing different positons for different chains and PDB Ids/
    for i in data.index:
        posOnPDB = {}
        uniprot_id = data.at[i, 'uniprotID']
        pdb_id = data.at[i, 'pdbID']
        pos = data.at[i, 'pos']
        wt = data.at[i, 'wt']
        data.at[i, 'AAonPDB'] = np.NaN
        data.at[i,'pdbinfo'] = pdb_id + data.at[i, 'chain']
        allMatchesForDP, full_ascaris = uniprot_pdb_residue_mapping(pdb_id, uniprot_id, save_path)
        for key, val in full_ascaris[data.at[i,'pdbinfo']].items():
            if int(key.split('_')[2]) == int(pos):
                data.loc[i, 'AAonPDB'] = val.split('_')[3]
                break

        if data.at[i, 'AAonPDB'] == wt:
            data.at[i, 'PDB_ALIGN_STATUS'] = 'aligned'
        else:
            data.at[i, 'PDB_ALIGN_STATUS'] = 'notAligned'
        keep = allMatchesForDP[data.at[i,'pdbinfo']]
        for pos in ast.literal_eval(data.at[i, 'POSITIONS']):
            try:
                if keep[str(pos)] != 'null':
                    posOnPDB[str(pos)] = keep[str(pos)]
                else:
                    pass
            except KeyError:
                pass

        data.at[i, 'MATCHDICT'] = str(posOnPDB)
    data = data.drop(columns=['POSITIONS'])
    return data


def processAnnotation(annot_positions):
    annot_positions = str(annot_positions).replace("'", '')
    annot_positions = str(annot_positions).replace('[', '')
    annot_positions = str(annot_positions).replace("]", '')
    positionList_perAnnotation = annot_positions.split(',')
    positionList_perAnnotation = [h.strip() for h in positionList_perAnnotation]
