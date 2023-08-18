### Datasets in the 'Paper Analyses' folder
The files in this directory concerns the various analyses conducted throughout the ASCARIS paper. They are mainly used to train-test machine learning (ML) models in the use-case application of ASCARIS for variant effect prediction. Datasets (ASCARIS feature vectors created for various SAV datasets/benchmarks) that are created using Alphafold proteins and PDB structures can be found under 2 different folders (folder names are "PDB" and "Alphafold", as well). In these folders, each file has a _raw and _imputed version. In the latter one, empty values are imputed with the mean values of each property in the training set. Imputed versions are used in the VEP analysis. Imputation and structural source info (PDB or Alphafold) are given at the end of each file name, hence is not written separately here, in order not to clutter the list. There is a third folder named "ML-based-VEP_scripts" which contain MATLAB scripts written to train and test VEP models.

- **training_uptodate_full.txt** : Full model training dataset obtained from UniProt, PMD and ClinVar. This dataset was used to produce the results given in Table 2 of the paper. This dataset has also been used to extract statistics given in Figure 3 and 4 of the paper.
- **training_uptodate_full_2014selected** : 2014 subset of the model training dataset.
- **training_uptodate_full_2014selected_wo_mt** : 2014 subset of the model training dataset, test datapoints from MutationTaster are removed. This dataset was used to produce the results given in Table 3 of the paper.
- **training_uptodate_full_2014selected_wo_psnp** : 2014 subset of the model training dataset, test datapoints from PredictSNP are removed. This dataset was used to produce the results given in Table S6 of the paper.
- **training_uptodate_full_2014selected_wo_swiss** : 2014 subset of the model training dataset, test datapoints from SwissVar are removed. This dataset was used to produce the results given in Table S6 of the paper.
- **training_uptodate_full_2014selected_wo_varibench** : 2014 subset of the model training set, test datapoints from Varibench are removed. This dataset was used to produce the results given in Table S6 of the paper.
- **training_uptodate_full_pdb_imputed_wo3genes** : Imputed PDB model training dataset; without datapoints from BRCA1, P53 and CALM1 genes, to be used in the DMS benchmark. File is zipped due to size limitation. This dataset was used to produce the results given in Figure 5 of the paper.
- **training_uptodate_full_alphafold_imputed_wo3genes** : Imputed AlphaFold model training dataset; without datapoints from BRCA1, P53 and CALM1 genes, to be used in the DMS benchmark. File is zipped due to size limitation. This dataset was used to produce the results given in Figure 5 of the paper.
- **dms_test_pdb_brca1** : Model test dataset created for benchmarking BRCA1 variations in the DMS benchmark. This dataset was used to produce the results given in Figure 5 of the paper.
- **dms_test_pdb_calm1** : Model test dataset created for benchmarking CALM1 variations in the DMS benchmark. This dataset was used to produce the results given in Figure 5 of the paper.
- **dms_test_pdb_p53** : Model test dataset created for benchmarking P53 variations in the DMS benchmark. This dataset was used to produce the results given in Figure 5 of the paper.
- **mutationtaster_test** : Model test dataset for the MutationTaster benchmark. This dataset was used to produce the results given in Table 3 of the paper.
- **psnp_test** : Model test dataset for the PredictSNP benchmark. This dataset was used to produce the results given in Table S6 of the paper.
- **swiss_test** : Model test dataset for the SwissVar benchmark. This dataset was used to produce the results given in Table S6 of the paper.
- **varibench_test** : Model test dataset for the VariBench benchmark. This dataset was used to produce the results given in Table S6 of the paper.


### Reproducible run of ASCARIS to generate the SAV representation datasets used in different parts of the study

