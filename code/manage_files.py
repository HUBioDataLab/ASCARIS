import os
import streamlit as st
from pathlib import Path
def manage_files(mode):
    if mode== 1:
        path_to_input_files = Path('input_files')
        path_to_domains = path_to_input_files / 'domains.txt'
        swiss_model_path = path_to_input_files / 'INDEX.json'
        fisher_path = path_to_input_files / 'significant_domains.txt'
        path_to_interfaces = path_to_input_files / 'H_sapiens_interfacesHQ.txt'

        os.makedirs('out_files', exist_ok=True)
        path_to_output_files = Path('out_files/pdb')
        os.makedirs(path_to_output_files / 'pdb_structures/', exist_ok=True)
        os.makedirs(path_to_output_files / 'alignment_files/', exist_ok=True)
        os.makedirs(path_to_output_files / 'swissmodel_structures/', exist_ok=True)
        os.makedirs(path_to_output_files / 'modbase_structures/', exist_ok=True)
        os.makedirs(path_to_output_files / 'modbase_structures_individual/', exist_ok=True)
        os.makedirs(path_to_output_files / 'freesasa_files/', exist_ok=True)
        os.makedirs(path_to_output_files / '3D_alignment/', exist_ok=True)
        path_to_alignment_files = path_to_output_files / 'alignment_files'
        path_3D_alignment = path_to_output_files / '3D_alignment'
        path_to_freesasa = path_to_output_files / 'freesasa_files'
        buffer = path_to_output_files / 'file_buffer.txt'
        outpath = path_to_output_files / 'feature_vector.txt'

        return path_to_input_files, path_to_output_files, path_to_domains,fisher_path, path_to_interfaces, buffer

    elif mode == 2:
        path_to_input_files = Path('input_files')
        path_to_domains = path_to_input_files / 'domains.txt'
        fisher_path = path_to_input_files / 'significant_domains.txt'
        alphafold_summary = path_to_input_files / 'alphafold_summary.txt'
        path_to_interfaces = path_to_input_files / 'H_sapiens_interfacesHQ.txt'
        # Unzip before using
        alphafold_path = Path(path_to_input_files/'alphafold_structures')

        os.makedirs('out_files', exist_ok=True)
        path_to_output_files = Path('out_files/alphafold')
        os.makedirs(path_to_output_files, exist_ok=True)
        os.makedirs(path_to_output_files / 'freesasa_files', exist_ok=True)
        os.makedirs(path_to_output_files / 'alignment_files', exist_ok=True)
        os.makedirs(path_to_output_files / '3D_alignment', exist_ok=True)

        return path_to_input_files,path_to_output_files, path_to_domains, fisher_path, path_to_interfaces, alphafold_path, alphafold_summary
