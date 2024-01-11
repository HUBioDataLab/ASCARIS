import glob
import ssbio.utils
import subprocess
import ssbio
import os.path as op
import os
from pathlib import Path
import gzip
import shutil
import streamlit as st
from utils import *


def run_freesasa(infile, outfile, include_hetatms=True, outdir=None, force_rerun=False, file_type = 'gzip'):
    if not outdir:
        outdir = ''
    outfile = op.join(outdir, outfile)
    if file_type == 'pdb':
        if ssbio.utils.force_rerun(flag=force_rerun, outfile=outfile):
            if include_hetatms:
                shell_command = 'freesasa --format=rsa --hetatm {} -o {}'.format(infile, outfile)
            else:
                shell_command = 'freesasa --format=rsa {} -o {}'.format(infile, outfile)
            command = subprocess.Popen(shell_command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       shell=True)
            out, err = command.communicate()
    elif file_type == 'gzip':
        with gzip.open(infile, 'rb') as f_in:
            with open('file_temp.pdb', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        infile = 'file_temp.pdb'

        if ssbio.utils.force_rerun(flag=force_rerun, outfile=outfile):
            if include_hetatms:
                shell_command = 'freesasa --format=rsa --hetatm {} -o {}'.format(infile, outfile)
            else:
                shell_command = 'freesasa --format=rsa {} -o {}'.format(infile, outfile)
            command = subprocess.Popen(shell_command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       shell=True)
            out, err = command.communicate()
    return outfile

def calculate_freesasa(ID, model_num, existing_free_sasa, path_to_input,path_to_output_files, file_type = 'gzip'):
    print('Calculating surface area...\n')
    file_base = str(Path(path_to_input / '*'))
    file_str = glob.glob(file_base)[0].split('-')[-1].split('.')[0]
    if file_type == 'gzip':
        if ID not in existing_free_sasa:
            fullID = f'AF-{ID}-F{model_num}-{file_str }.pdb.gz'
            run_freesasa(Path(path_to_input / fullID),
                         Path(path_to_output_files / f'freesasa_files/{fullID}.txt'), include_hetatms=True,
                         outdir=None, force_rerun=False)
    elif file_type == 'pdb':
        if ID not in existing_free_sasa:
            fullID = f'AF-{ID}-F{model_num}-model_v1.pdb'
            run_freesasa(Path(path_to_input / fullID),
                         Path(path_to_output_files / f'freesasa_files/{fullID}.txt'), include_hetatms=True,
                         outdir=None, force_rerun=False)

def sasa(uniprotID, sasa_pos, wt, mode, filename, path_to_output_files, file_type = 'gzip'):
    if mode == 1:
        files = open(filename, 'r')
        file = files.readlines()
        for k in file:
            if str(k.strip()[10:13].strip()) == str(sasa_pos):
                residue = str(k[4:7].strip())
                if wt == threeToOne(residue):
                    sasa = str(k[22:28]).strip('\n')
                    return (sasa)
                elif wt != threeToOne(residue):
                    sasa = str(k[22:28]).strip('\n') + '*'
                    return (sasa)
                else:
                    return 'nan'
    if mode == 2:
        if sasa_pos != np.NaN:
            sasa = 'nan'
            if file_type == 'pdb':
                for filename in list(Path(path_to_output_files / 'freesasa_files').glob("*")):
                    fname = list(filter(None, filename.split('.'))).split('/')[-1].upper()
                    if uniprotID == fname:
                        files = open(filename, 'r')
                        file = files.readlines()
                        for k in file:
                            if k.strip()[10:13] == sasa_pos:
                                residue = str(k[4:7].strip())
                                if wt == threeToOne(residue):
                                    sasa = str(k[22:28]).strip('\n')
                                elif wt != threeToOne(residue):
                                    sasa = str(k[22:28]).strip('\n') + '*'

                return sasa
            elif file_type == 'gzip':
                for filename in  list(Path(path_to_output_files / 'freesasa_files').glob("*")):
                    fname = list(filter(None, str(filename).split('.')))[0].split('/')[-1].split('-')[1].upper()

                    if uniprotID == fname:
                        files = open(filename, 'r')
                        file = files.readlines()
                        for k in file:
                            if str(k.strip()[10:13]).strip() == str(sasa_pos):
                                residue = str(k[4:7].strip())
                                if wt == threeToOne(residue):
                                    sasa = str(k[22:28]).strip('\n')
                                elif wt != threeToOne(residue):
                                    sasa = str(k[22:28]).strip('\n') + '*'
                                else:
                                    sasa = 'nan'

                return sasa
        else:
            sasa = 'nan'
            return sasa
