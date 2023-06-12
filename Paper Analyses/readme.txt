The files in these folders concerns the various analyses conducted throughout the ASCARIS
paper. They are used to create machine learning models. Each file has a _raw and _imputed 
version. In the latter one the empty values are imputed with the mean values of each 
property obtained from the main training set. Feature vectors that are created using 
Alphafold proteins and PDB structures can be found under related folders. Imputation and
structure sources are given at the end of each file name, hence is not written separately 
here.


- **training_uptodate_full** : Full training dataset obtained from UniProt, PMD and ClinVar.
- **training_uptodate_full_2014selected** : 2014 subset of the training set.
- **training_uptodate_full_2014selected_wo_mt** : 2014 subset of the training set, test 
datapoints from MutationTaster removed.
- **training_uptodate_full_2014selected_wo_psnp** : 2014 subset of the training set, test 
datapoints from PredictSNP removed.
- **training_uptodate_full_2014selected_wo_swiss** : 2014 subset of the training set, test 
datapoints from SwissVar removed.
- **training_uptodate_full_2014selected_wo_varibench** : 2014 subset of the training set, test 
datapoints from Varibench removed.
- **training_uptodate_full_pdb_imputed_wo3genes** : Imputed PDB training feature vector without 
datapoints from BRCA1, P53 and CALM1. File is zipped due to size limitations.
- **training_uptodate_full_pdb_raw_wo3genes** : Non-imputed PDB training feature vector without 
datapoints from BRCA1, P53 and CALM1.
- **training_uptodate_full_alphafold_imputed_wo3genes** : Imputed AlphaFold training feature 
vector without datapoints from BRCA1, P53 and CALM1. File is zipped due to size limitations.
- **training_uptodate_full_alphafold_raw_wo3genes** : Non-imputed AlphaFold training feature 
vector without datapoints from BRCA1, P53 and CALM1.
- **dms_test_pdb_brca1_imputed** : Feature vector created for benchmarking BRCA1 variations. 
- **dms_test_pdb_calm1** : Feature vector created for benchmarking CALM1 variations. .
- **dms_test_pdb_p53** : Feature vector created for benchmarking P53 variations.
- **mutationtaster_test** : Benchmark set obtained from MutationTaster data.
- **psnp_test** : Benchmark set obtained from PredictSNP database.
- **swiss_test** : Benchmark set obtained from SwissVar database.
- **varibench_test** : Benchmark set obtained from VariBench database.



