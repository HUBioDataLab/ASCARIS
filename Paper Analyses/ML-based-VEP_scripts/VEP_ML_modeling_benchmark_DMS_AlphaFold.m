% DMS Benchmark (AlphaFold)

% Data loading:

opts = detectImportOptions('DMS_benchmark_modeling/training_uptodate_full_alphafold_imputed_wo6genesLabelled.txt');
opts = setvartype(opts,opts.VariableNames(1,[10:11 14:44]),'char');
opts = setvartype(opts,opts.VariableNames(1,[6:9 12:13 45:75]),'double');

T_full_DMS_train_raw = readtable('DMS_benchmark_modeling/training_uptodate_full_alphafold_imputed_wo6genesLabelled.txt',opts);
neut_delet_DMS_train = table2array(T_full_DMS_train_raw(:,75));
T_full_DMS_train = T_full_DMS_train_raw(:,[6:9 11:74]);

neut_delet_DMS_test_raw = readtable('DMS_benchmark_modeling/test_datapoints_wLabel.txt');

T_full_DMS_test_brca1_raw = readtable('DMS_benchmark_modeling/test_featurevector_alphafold_brca1.txt',opts);
T_full_DMS_test_brca1_raw_sort = sortrows(T_full_DMS_test_brca1_raw,'position');
neut_delet_DMS_test_brca1 = neut_delet_DMS_test_raw.Class(1:834);
T_full_DMS_test_brca1 = T_full_DMS_test_brca1_raw_sort(:,[6:9 11:74]);

T_full_DMS_test_calm1_raw = readtable('DMS_benchmark_modeling/test_featurevector_alphafold_calm1.txt',opts);
T_full_DMS_test_calm1_raw_sort = sortrows(T_full_DMS_test_calm1_raw,'position');
neut_delet_DMS_test_calm1 = neut_delet_DMS_test_raw.Class(835:864);
T_full_DMS_test_calm1 = T_full_DMS_test_calm1_raw_sort(:,[6:9 11:74]);

T_full_DMS_test_hras_raw = readtable('DMS_benchmark_modeling/test_featurevector_alphafold_hras.txt',opts);
T_full_DMS_test_hras_raw_sort = sortrows(T_full_DMS_test_hras_raw,'position');
neut_delet_DMS_test_hras = neut_delet_DMS_test_raw.Class(865:975);
T_full_DMS_test_hras = T_full_DMS_test_hras_raw_sort(:,[6:9 11:74]);

T_full_DMS_test_p53_raw = readtable('DMS_benchmark_modeling/test_featurevector_alphafold_p53.txt',opts);
T_full_DMS_test_p53_raw_sort = sortrows(T_full_DMS_test_p53_raw,'position');
neut_delet_DMS_test_p53 = neut_delet_DMS_test_raw.Class(976:1350);
T_full_DMS_test_p53 = T_full_DMS_test_p53_raw_sort(:,[6:9 11:74]);

T_full_DMS_test_pten_raw = readtable('DMS_benchmark_modeling/test_featurevector_alphafold_pten.txt',opts);
T_full_DMS_test_pten_raw_sort = sortrows(T_full_DMS_test_pten_raw,'position');
neut_delet_DMS_test_pten = neut_delet_DMS_test_raw.Class(1351:1536);
T_full_DMS_test_pten = T_full_DMS_test_pten_raw_sort(:,[6:9 11:74]);

T_full_DMS_test_tpk1_raw = readtable('DMS_benchmark_modeling/test_featurevector_alphafold_tpk1.txt',opts);
T_full_DMS_test_tpk1_raw_sort = sortrows(T_full_DMS_test_tpk1_raw,'position');
neut_delet_DMS_test_tpk1 = neut_delet_DMS_test_raw.Class(1537:end);
T_full_DMS_test_tpk1 = T_full_DMS_test_tpk1_raw_sort(:,[6:9 11:74]);


save DMS_benchmark_modeling/T_all_DMS_alphafold.mat T_full_DMS_train neut_delet_DMS_train neut_delet_DMS_test_raw T_full_DMS_test_brca1 neut_delet_DMS_test_brca1 T_full_DMS_test_calm1 neut_delet_DMS_test_calm1 T_full_DMS_test_hras neut_delet_DMS_test_hras T_full_DMS_test_p53 neut_delet_DMS_test_p53 T_full_DMS_test_pten neut_delet_DMS_test_pten T_full_DMS_test_tpk1 neut_delet_DMS_test_tpk1



% Training and testing:

t = templateTree('MaxNumSplits',(size(T_full_DMS_train,1)-1),'NumVariablesToSample',8);
Mdl_full_cval_DMS=fitcensemble(T_full_DMS_train,neut_delet_DMS_train,'Method','Bag','CrossVal','on','KFold',5,'NumLearningCycles',300,'Learners',t);
[validationPredictions,validationScores]=kfoldPredict(Mdl_full_cval_DMS);
confmat=confusionmat(Mdl_full_cval_DMS.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_full_cval_DMS.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_full_cval_DMS.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_cval=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);


t = templateTree('MaxNumSplits',(size(T_full_DMS_train,1)-1),'NumVariablesToSample',8);
Mdl_full_test_DMS=fitcensemble(T_full_DMS_train,neut_delet_DMS_train,'Method','Bag','NumLearningCycles',300,'Learners',t);


[testPredictions,testScores]=predict(Mdl_full_test_DMS,T_full_DMS_test_brca1);
confmat=confusionmat(neut_delet_DMS_test_brca1,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_DMS_test_brca1==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_DMS_test_brca1,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_test=table({'brca1'},AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

[testPredictions,testScores]=predict(Mdl_full_test_DMS,T_full_DMS_test_calm1);
confmat=confusionmat(neut_delet_DMS_test_calm1,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_DMS_test_calm1==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_DMS_test_calm1,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_test(end+1,:)=table({'calm1'},AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

[testPredictions,testScores]=predict(Mdl_full_test_DMS,T_full_DMS_test_hras);
confmat=confusionmat(neut_delet_DMS_test_hras,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_DMS_test_hras==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_DMS_test_hras,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_test(end+1,:)=table({'hras'},AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

[testPredictions,testScores]=predict(Mdl_full_test_DMS,T_full_DMS_test_p53);
confmat=confusionmat(neut_delet_DMS_test_p53,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_DMS_test_p53==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_DMS_test_p53,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_test(end+1,:)=table({'p53'},AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

[testPredictions,testScores]=predict(Mdl_full_test_DMS,T_full_DMS_test_pten);
confmat=confusionmat(neut_delet_DMS_test_pten,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_DMS_test_pten==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_DMS_test_pten,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_test(end+1,:)=table({'pten'},AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

[testPredictions,testScores]=predict(Mdl_full_test_DMS,T_full_DMS_test_tpk1);
confmat=confusionmat(neut_delet_DMS_test_tpk1,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_DMS_test_tpk1==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_DMS_test_tpk1,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_DMS_test(end+1,:)=table({'tpk1'},AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);


save DMS_benchmark_modeling/Mdl_full_cval_DMS_alphafold.mat Mdl_full_cval_DMS 
save DMS_benchmark_modeling/Mdl_full_test_DMS_alphafold.mat Mdl_full_test_DMS
save DMS_benchmark_modeling/Perf_tables_DMS_alphafold.mat Perf_table_DMS_cval Perf_table_DMS_test




% Training the finalized model (Hyper-parameter optimization):
% 
% n = size(T_full_DMS_train,1);
% m = floor(log(n - 1)/log(3));
% maxNumSplits = 3.^(0:m);
% maxNumSplits = [maxNumSplits(8) size(T_full_DMS_train,1)-1];
% numMNS = numel(maxNumSplits);
% numTrees = [300 500];
% numT = numel(numTrees);
% numvartosamp = [8 24 size(T_full_DMS_train,2)];
% numV = numel(numvartosamp);
% Mdl_rf_DMS_hypopt = cell(numT,numMNS,numV);
% AUC=zeros(numT*numV*numMNS,1);Recall=zeros(numT*numV*numMNS,1);Precision=zeros(numT*numV*numMNS,1);F1score=zeros(numT*numV*numMNS,1);Accuracy=zeros(numT*numV*numMNS,1);MCC=zeros(numT*numV*numMNS,1);TP=zeros(numT*numV*numMNS,1);FP=zeros(numT*numV*numMNS,1);FN=zeros(numT*numV*numMNS,1);TN=zeros(numT*numV*numMNS,1);
% NumVartoSam=repmat(numvartosamp',numT*numMNS,1);
% maxNumSplit=repmat([repmat(maxNumSplits(1,1),numV,1);repmat(maxNumSplits(1,2),numV,1)],numT,1);
% numTree=repmat(numTrees,numV*numMNS,1);numTree=reshape(numTree,[size(numTree,1)*size(numTree,2),1]);
% Perf_table_DMS_test_hyperpar=table(numTree,maxNumSplit,NumVartoSam,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
% to=0;
% for i = 1:numT
%     for j = 1:numMNS
%         for k = 1:numV
%             to=to+1;
%             t = templateTree('MaxNumSplits',maxNumSplits(j),'NumVariablesToSample',numvartosamp(k));
%             disp(['numTrees: ', num2str(numTrees(i)), ', maxNumSplit: ', num2str(maxNumSplits(j)), ', NumVariablesToSample: ', num2str(numvartosamp(k))])
%             Mdl_rf_DMS_hypopt{i,j,k} = fitcensemble(T_full_DMS_train,neut_delet_DMS_train,'Method','Bag','NumLearningCycles',numTrees(i),'Learners',t);
%             [testPredictions,testScores]=predict(Mdl_rf_DMS_hypopt{i,j,k},T_full_DMS_test);
%             confmat=confusionmat(neut_delet_DMS_test,testPredictions);
%             confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
%             [~,~,~,AUC]=perfcurve(neut_delet_DMS_test,testScores(:,2),1);
%             TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
%             Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
%             Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
%             Perf_table_DMS_test_hyperpar(to,4:13)=Perf_table_temp;
%         end
%     end
% end
% (the best model scored MCC: ... which is from the previous run, not in this grid search and the hyperparamter values are; ..., MaxNumSplits: ..., NumVariablesToSample: ...)
% 
% save DMS_benchmark_modeling/Mdl_DMS_hypopt.mat Mdl_rf_DMS_hypopt
% save DMS_benchmark_modeling/Perf_tables_DMS_hyperpar.mat Perf_table_DMS_test_hyperpar



