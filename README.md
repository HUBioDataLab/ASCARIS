# ASCARIS: Positional Feature Annotation and Protein Structure-Based Representation of Single Amino Acid Variations

## Abstract 

Genomic variations may cause deleterious effects on protein functionality and perturb biological processes. Elucidating the effects of variations is critical for developing novel treatment strategies for diseases of genetic origin. 

ASCARIS (Annotation and StruCture-bAsed RepresentatIon of Single amino acid variations) is a tool for the featurization (i.e., quantitative representation) of SAVs, which could be used for a variety of purposes, such as predicting their functional effects or building multi-omics-based integrative models. ASCARIS utilizes the correspondence between the location of the SAV on the sequence and 30 different types of positional feature annotations (e.g., active/lipidation/glycosylation sites; calcium/metal/DNA binding, inter/transmembrane regions, etc.) from UniProt, along with structural features and the change in physicochemical properties, using models from PDB and AlphaFold-DB. It constructs a 74-dimensional feature set to represent a given SAV. 

For more information on the construction of the feature vector, the statistical analysis of different features, and machine learning models trained on the feature vectors that predicts the effect of SAVs, please refer to the main article:
Cankara, F., & Dogan, T. (2022). ASCARIS: Positional Feature Annotation and Protein Structure-Based Representation of Single Amino Acid Variations. *bioRxiv*, 514934v1. [Link](https://www.biorxiv.org/content/10.1101/2022.11.03.514934v1)

<p align="center">
 
<img width="919" alt="ASCARIS_Overall_Workflow" src="https://github.com/HUBioDataLab/ASCARIS/assets/26777185/da563283-d696-45a6-adf0-9c97f8d04018">


<img width="1087" alt="ASCARIS_Overall_Workflow" src="https://user-images.githubusercontent.com/13165170/198850423-5d50bde2-f9dd-4600-baae-65830afdb57f.png">

 </p>
 
 
## Development and Dependencies

- [Python 3.7.3](https://www.python.org/downloads/release/python-373/)
- [Pandas 1.1.4](https://pandas.pydata.org/pandas-docs/version/1.1.4/getting_started/install.html)
- [Numpy 1.19.5](https://numpy.org/devdocs/release/1.19.5-notes.html)
- [Ssbio 0.9.9.8.post1](https://pypi.org/project/ssbio/)
- [Freesasa 2.0.3.post7](https://pypi.org/project/freesasa/2.0.3.post7/)
- [Requests 2.22.0](https://pypi.org/project/requests/)
- [Biopython 1.78](https://biopython.org/docs/1.78/api/Bio.html)

## Descriptions of folders and files in the ASCARIS repository 

### Paper Analyses

All files used to create feature vectors that are used in the main ASCARIS paper are presented under **Paper Analyses** directory. PDB and AlphaFold contains files that are created using PDB structures and AlphaFold models, respectively.

### Input Files 

Files that are necessary to run the ASCARIS tool are found under **input_files** folder.

- **swissmodel_structures.txt.zip** : Includes summary file for Swiss-Model structures. Swiss-Model summary (INDEX-metadata) files are downloaded separately for each organism from https://swissmodel.expasy.org/repository, and merged into a single file by running create_swissmodelSummary.py code file. Generated file is uploaded to GitHub as a zip file, thus **it must be unzipped to input_files folder prior to usage**. Alternatively it can be downloaded [here](https://drive.google.com/drive/u/1/folders/1pJyXcguupyGggl25fzbRWwwqC6qUbDka). If needed, the user can create an updated file by running script **create_swissmodelSummary.py** in the directory in which newly downloaded meta-data is found. Relevant output file will be created under /input_files.
```
cd ASCARIS
python3 code/create_swissmodelSummary.py -folder_name folder_to_meta_data
```

- **domains.txt** : Includes InterPro domains simplified as in the following order *(tab separated)* --> 
  [uniprotID      domainID        domainStartPosition     domainEndPosition]
- **significant_domains** :  Selected domains from *domains.txt* file according to Fisher's Exact Test result. Fisher's Exact Test applied to all domains in the training test to assess their significance with respect to the the deleteriousness outcome. p_values is chosen as 0.01.
- **H_sapiens_interfacesHQ.txt** :  High confidence interfaces downloaded from [Interactome Insider](http://interactomeinsider.yulab.org/downloads.html) for *Homo sapiens*
- **alphafold_structures** : This folder contains [AlphaFold Human proteome predictions](http://ftp.ebi.ac.uk/pub/databases/alphafold/latest/). **Please download the '.tar' file to the input_files folder** and **run get_alphafoldStructures.py** to untar the structures and create alphafold_summary file. The current folder in this repository contains 100 AlphaFold model files for demo purposes, hence the users need to download the complete set of AlphaFold structures prior to running ASCARIS. 
```
cd ASCARIS
python3 code/get_alphafoldStructures.py -file_name UP000005640_9606_HUMAN.tar
```
- **alphafold_summary**: Processed data for AlphaFold structures. Includes protein identifier, chain id, sequence, model count for each entry.

## ASCARIS Usage

This section intends to guide the users on how to run ASCARIS. 

Please unzip required files prior to running the code as described in the **Input Files** section.
ASCARIS can be run in three ways.

1. Run ASCARIS for only one datapoint:

```
python3 code/main.py -s 1 -i P13637-T-613-M -impute True
```
2. Run ASCARIS for more than one datapoints:
```
python3 code/main.py -s 2 -i 'P13637-T-613-M, Q9Y4W6-N-432-T, Q9Y4W6-N-432-T' impute False
```

3. Run ASCARIS on a tab-separated file containing SAV information. Please see sample_input.txt for the format.
```
python3 code/main.py -s 2 -i input_files/sample_input.txt
```
### Input Arguments

-s :  selection for input structure data. (1: Use PDB-ModBase-SwissModel structures, 2: Use AlphaFold Structures) </br>

-i :  input option. Enter datapoint to predict or input file name in the following form:</br>
- *Option 1: Comma-separated list of idenfiers (UniProt ID-wt residue-position-mutated residue (e.g. Q9Y4W6-N-432-T or Q9Y4W6-N-432-T, Q9Y4W6-N-432-T))*  
- *Option 2: Enter tab-separated file path*

-impute :  imputation of NaN values. Imputation values are median values of corresponding columns. Default True </br>


### Sample Output 

This folder contains demo results for ASCARIS with **sample_input.txt** file. Example input file format is shown below. Columns represent UniProt ID of the protein, wild type amino acid, position of the amino acid change and mutated amino acid, respectively. Input file must be given **without** a header.


```
P12694	C	264	W
P13637	T	613	M
P05067	I	716	V
P41180	E	604	K
P08123	G	646	C
P06731	I	80	V
P29474	D	298	E
Q16363	Y	498	H
P23560	V	66	M
Q00889	H	85	D
```

Upon running ASCARIS, **out_files** folder will be created. Depending on the selected arguments, two type of sub-folders (PDB and AlphaFold) will be created. 

*__If the user wants to run ASCARIS using PDB-ModBase-SwissModel structures, the argument -s should be set to 1:__*

```
python3 code/main.py -s 1 -i input_files/sample_input.txt
```

Upon running the line above, the folllowing files will be generated: 

- **pdb/pdb_structures** : Contains downloaded structure files from PDB for input proteins when applicable. If the user has a folder wherein PDB structures are stored, please change the name of that folder to pdb_structures and the extension of files to '.txt' to decrease run time. 
- **pdb/swissmodel_structures** : Contains downloaded model files from SwissModel for input proteins when applicable.
- **pdb/modbase_structures** : Contains downloaded model files from ModBase for input proteins when applicable. Each file contains all models related to one protein.
- **pdb/modbase_structures_individual** : Contains downloaded model files from ModBase for input proteins when applicable. Each file contains individual models related to one protein.
- **pdb/alignment_files** : Contains alignment files of protein sequences. 
- **pdb/3D_alignment** : Contains alignment files of structure files. This step is performed in order to avoid missing residues in the PDB files.
- **pdb/sasa_files** : Contains calculated solvent accessible surface area values for each data point.
- **pdb/feature_vector.txt** : Final feature vector file.
- **pdb/log.txt** : Log file




*__If the user wants to run ASCARIS using Alphafold models, the argument -s should be set to 2:__*

```
python3 code/main.py -s 2 -i input_files/sample_input.txt
```

Upon running the line above, the folllowing files will be generated: 

- **alphafold/alignment_files** : Contains alignment of UniProt sequence files.
- **alphafold/3D_alignment** :  Contains alignment of UniProt sequence files to PDB sequence files.
- **alphafold/sasa_files** : Contains calculated solvent accessible surface area values for each data point.
- **alphafold/featurevector_alphafold.txt** : Final feature vector file.
- **alphafold/log.txt** : Log file


## Description of the Dimensions of the Output Representations

<img width="1284" alt="ASCARIS_Representation_Dimensions" src="https://user-images.githubusercontent.com/13165170/198850505-2a493c6a-a55d-43f1-af81-1c2fff5ac7ed.png">

In ASCARIS representations, dimensions 1-5 correspond to datapoint identifier, 6-9 correspond to physicochemical property values, 10-12 correspond to domain-related information, 13-14 correspond to information regarding variation's position on the protein (both the sasa value and the categorization), 15-44 correspond to binary correspondence between variations and different types of positional annotations (1 dimension for each annotation type), 45-74 correspond to spatial (Euclidian) distances between variations and different types of positional annotations (1 dimension for each annotation type).


| Order of dimension | Column name in the output file  | Description |  Source | 
| ------------- | ------------- | ------------- | ------------- |
| 1 | prot_uniprotAcc | UniProt accession | Metadata obtained from UniProtKB/Swiss-Prot |
| 2 | wt_residue | Wild type residue | Data obtained from UniProtKB/Swiss-Prot (humsavar), ClinVar, PMD |
| 3 | mut_residue | Mutated residue | Data obtained from UniProtKB/Swiss-Prot (humsavar), ClinVar, PMD |
| 4 | position | Variation position | Data obtained from UniProtKB/Swiss-Prot (humsavar), ClinVar, PMD |
| 5 | meta_merged | Datapoint identifier (UniProt accession-WT Residue-VariationPosition-Mutated Residue) | - |
| 6 | composition | Change in composition values upon the occurrence of variation. Composition is defined as the atomic weight ratio of hetero (non-carbon) elements in end groups or rings to carbons in the side chain. | Literature |
| 7 | polarity | Change in polarity values upon variation. | Literature |
| 8 | volume | Change in volume values upon variation. | Literature |
| 9 | granthamScore | Change in Grantham scores (the combination of composition, polarity and volume) values upon variation. | Literature |
| 10 | domains_all | InterPro Domain IDs of all domains found in the dataset  | Data obtained from InterPro |
| 11 | domains_sig | InterPro Domain IDs of significant domains in the dataset. Domains that are not found to be significant in Fisher's Exact Test are labelled as "NULL". | Data obtained from InterPro |
| 12 | domains_3Ddist | Shortest Euclidian distance between the domain and the variation site. | A newly engineered feature (data obtained from PDB/AlphaFold and InterPro) |
| 13 | sasa | Solvent accessible surface area values. | FreeSASA |
| 14 | location_3state | Caterozied location of the variation in the structure: surface, core or interface. | FreeSASA, InteractomeInsider |
| 15-44 |disulfide_bin, intMet_bin,intramembrane_bin, naturalVariant_bin, dnaBinding_bin, activeSite_bin, nucleotideBinding_bin, lipidation_bin, site_bin, transmembrane_bin, crosslink_bin, mutagenesis_bin, strand_bin, helix_bin, turn_bin, metalBinding_bin, repeat_bin, caBinding_bin, topologicalDomain_bin, bindingSite_bin, region_bin, signalPeptide_bin, modifiedResidue_bin, zincFinger_bin, motif_bin, coiledCoil_bin, peptide_bin, transitPeptide_bin, glycosylation_bin, propeptide_bin | Positional sequence annotations, binary correspondence-based (30 different types of annotations, each one on a different dimension). Categories: 0: annotatation does not exist on the protein, 1: annotation is presented, but the variation is not on the annotated site, 2: variation is on the annotated site. | Newly engineered features (data obtained from UniProtKB) |
| 45-74 |disulfide_dist, intMet_dist, intramembrane_dist, naturalVariant_dist, dnaBinding_dist, activeSite_dist, nucleotideBinding_dist, lipidation_dist, site_dist, transmembrane_dist, crosslink_dist, mutagenesis_dist, strand_dist, helix_dist, turn_dist, metalBinding_dist, repeat_dist, caBinding_dist, topologicalDomain_dist, bindingSite_dist, region_dist, signalPeptide_dist, modifiedResidue_dist, zincFinger_dist, motif_dist, coiledCoil_dist, peptide_dist, transitPeptide_dist, glycosylation_dist, propeptide_dist | Positional sequence annotations, distance-based (the spatial distance between the annotated residue and the mutated residue, in the protein structure, for 30 different types of annotations, each one on a different dimension), in terms of Angstroms. | Newly engineered features (data obtained from PDB/AlphaFold and UniProtKB) |

## Please refer for more information:

Cankara, F., & Dogan, T. (2022). ASCARIS: Positional Feature Annotation and Protein Structure-Based Representation of Single Amino Acid Variations. *bioRxiv*, 514934v1. [Link](https://www.biorxiv.org/content/10.1101/2022.11.03.514934v1)

## License
Copyright (C) 2022 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
