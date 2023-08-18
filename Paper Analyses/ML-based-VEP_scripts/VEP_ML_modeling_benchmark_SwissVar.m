% SwissVar Benchmark

% Data loading:

opts = detectImportOptions('Paper_files/originaldata/training_uptodate_full_2014selected_swiss.txt');
opts = setvartype(opts,opts.VariableNames(1,[13:14 17:47]),'char');
opts = setvartype(opts,opts.VariableNames(1,[9:12 15:16 48:77]),'double');

T_benchmark_sv = readtable('Paper_files/originaldata/training_uptodate_full_2014selected_swiss.txt',opts);
neut_delet_sv = table2array(T_benchmark_sv(:,78));
T_full_sv = T_benchmark_sv(:,[9:12 14:77]);

T_benchmark_sv_test = readtable('Paper_files/benchmark/swiss.txt',opts);
neut_delet_sv_test = table2array(T_benchmark_sv_test(:,78));
T_full_sv_test = T_benchmark_sv_test(:,[9:12 14:77]);

save SwissVar_benchmark_modeling/T_all_sv.mat T_full_sv neut_delet_sv T_full_sv_test neut_delet_sv_test T_benchmark_sv_test T_benchmark_sv



% Training and testing:

Mdl_full_cval_sv=fitcensemble(T_full_sv,neut_delet_sv,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_full_cval_sv);
confmat=confusionmat(Mdl_full_cval_sv.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_full_cval_sv.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_full_cval_sv.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_sv_cval=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

Mdl_full_test_sv=fitcensemble(T_full_sv,neut_delet_sv,'Method','Bag');
[testPredictions,testScores]=predict(Mdl_full_test_sv,T_full_sv_test);
confmat=confusionmat(neut_delet_sv_test,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_sv_test==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_sv_test,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_sv_test=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

save SwissVar_benchmark_modeling/Mdl_full_sv.mat Mdl_full_cval_sv Mdl_full_test_sv
save SwissVar_benchmark_modeling/Perf_tables_sv.mat Perf_table_sv_cval Perf_table_sv_test



% Training the finalized model (Hyper-parameter optimization):

n = size(T_full_sv,1);
m = floor(log(n - 1)/log(3));
maxNumSplits = 3.^(0:m);
maxNumSplits = [maxNumSplits([8]) size(T_full_sv,1)-1];
numMNS = numel(maxNumSplits);
numTrees = [150 300];
numT = numel(numTrees);
numvartosamp = [8 24 size(T_full_sv,2)];
numV = numel(numvartosamp);
Mdl_rf_sv_hypopt = cell(numT,numMNS,numV);
AUC=zeros(numT*numV*numMNS,1);Recall=zeros(numT*numV*numMNS,1);Precision=zeros(numT*numV*numMNS,1);F1score=zeros(numT*numV*numMNS,1);Accuracy=zeros(numT*numV*numMNS,1);MCC=zeros(numT*numV*numMNS,1);TP=zeros(numT*numV*numMNS,1);FP=zeros(numT*numV*numMNS,1);FN=zeros(numT*numV*numMNS,1);TN=zeros(numT*numV*numMNS,1);
NumVartoSam=repmat(numvartosamp',numT*numMNS,1);
maxNumSplit=repmat([repmat(maxNumSplits(1,1),numV,1);repmat(maxNumSplits(1,2),numV,1)],numT,1);
numTree=repmat(numTrees,numV*numMNS,1);numTree=reshape(numTree,[size(numTree,1)*size(numTree,2),1]);
Perf_table_sv_test_hyperpar=table(numTree,maxNumSplit,NumVartoSam,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
to=0;
for i = 1:numT
    for j = 1:numMNS
        for k = 1:numV
            to=to+1;
            t = templateTree('MaxNumSplits',maxNumSplits(j),'NumVariablesToSample',numvartosamp(k));
            disp(['numTrees: ', num2str(numTrees(i)), ', maxNumSplit: ', num2str(maxNumSplits(j)), ', NumVariablesToSample: ', num2str(numvartosamp(k))])
            Mdl_rf_sv_hypopt{i,j,k} = fitcensemble(T_full_sv,neut_delet_sv,'Method','Bag','NumLearningCycles',numTrees(i),'Learners',t);
            [testPredictions,testScores]=predict(Mdl_rf_sv_hypopt{i,j,k},T_full_sv_test);
            confmat=confusionmat(neut_delet_sv_test,testPredictions);
            confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
            [~,~,~,AUC]=perfcurve(neut_delet_sv_test,testScores(:,2),1);
            TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
            Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
            Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
            Perf_table_sv_test_hyperpar(to,4:13)=Perf_table_temp;
        end
    end
end

save SwissVar_benchmark_modeling/Mdl_sv_hypopt.mat Mdl_rf_sv_hypopt
save SwissVar_benchmark_modeling/Perf_tables_sv_hyperpar.mat Perf_table_sv_test_hyperpar

[testPredictions,testScores]=predict(Mdl_rf_sv_hypopt{1,2,1},T_full_sv_test);
confmat=confusionmat(neut_delet_sv_test,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
[~,~,~,AUC]=perfcurve(neut_delet_sv_test,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_sv_test=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
save SwissVar_benchmark_modeling/Perf_tables_sv.mat Perf_table_sv_cval Perf_table_sv_test



% Family-based performance calculation on the MT test dataset:

Family=({'Enzymes';'Membrane_receptors';'Transcription_factors';'Ion_channels';'Epigenetic_regulators';'Others';'Others_4';'Overall'});
AUC=zeros(8,1);NPV=zeros(8,1);Specificity=zeros(8,1);Recall=zeros(8,1);Precision=zeros(8,1);F1score=zeros(8,1);Accuracy=zeros(8,1);MCC=zeros(8,1);TP=zeros(8,1);FN=zeros(8,1);FP=zeros(8,1);TN=zeros(8,1);
Perf_table_sv_test_family=table(Family,AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

family_sv_test=table2array(T_benchmark_sv_test(:,8));
ind_enzyme=find(contains(family_sv_test,'Enzymes')==1);neut_delet_sv_test_fam{1,1}=neut_delet_sv_test(ind_enzyme,1);testPred_fam{1,1}=testPredictions(ind_enzyme,1);testScor_fam{1,1}=testScores(ind_enzyme,:);
ind_membrane=find(contains(family_sv_test,'Membrane')==1);neut_delet_sv_test_fam{2,1}=neut_delet_sv_test(ind_membrane,1);testPred_fam{2,1}=testPredictions(ind_membrane,1);testScor_fam{2,1}=testScores(ind_membrane,:);
ind_tfactor=find(contains(family_sv_test,'Transcription')==1);neut_delet_sv_test_fam{3,1}=neut_delet_sv_test(ind_tfactor,1);testPred_fam{3,1}=testPredictions(ind_tfactor,1);testScor_fam{3,1}=testScores(ind_tfactor,:);
ind_ion=find(contains(family_sv_test,'Ion')==1);neut_delet_sv_test_fam{4,1}=neut_delet_sv_test(ind_ion,1);testPred_fam{4,1}=testPredictions(ind_ion,1);testScor_fam{4,1}=testScores(ind_ion,:);
ind_epigen=find(contains(family_sv_test,'Epigenetic')==1);neut_delet_sv_test_fam{5,1}=neut_delet_sv_test(ind_epigen,1);testPred_fam{5,1}=testPredictions(ind_epigen,1);testScor_fam{5,1}=testScores(ind_epigen,:);
ind_others=find(contains(family_sv_test,'nan')==1);neut_delet_sv_test_fam{6,1}=neut_delet_sv_test(ind_others,1);testPred_fam{6,1}=testPredictions(ind_others,1);testScor_fam{6,1}=testScores(ind_others,:);
ind_others4=unique([ind_tfactor;ind_ion;ind_epigen;ind_others]);neut_delet_sv_test_fam{7,1}=neut_delet_sv_test(ind_others4,1);testPred_fam{7,1}=testPredictions(ind_others4,1);testScor_fam{7,1}=testScores(ind_others4,:);
neut_delet_sv_test_fam{8,1}=neut_delet_sv_test;testPred_fam{8,1}=testPredictions;testScor_fam{8,1}=testScores;

for i=1:8
    confmat=confusionmat(neut_delet_sv_test_fam{i,1},testPred_fam{i,1});confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];[~,~,~,AUC]=perfcurve(neut_delet_sv_test_fam{i,1},testScor_fam{i,1}(:,2),1);TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_temp=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
    Perf_table_sv_test_family(i,2:13)=Perf_table_temp;
end

save SwissVar_benchmark_modeling/Perf_tables_sv_family.mat Perf_table_sv_test_family



% Other methods' performance calculations:

Methods=({'Logit';'Logit+';'Condel';'Condel+';'FatHMM-W';'FatHMM-U';'LRT';'SIFT';'MutationTaster';'MutationAssessor';'PolyPhen2'});

sv_input_data = readtable('Paper_files/benchmark/inputs_to_create_featurevector_benchmark/swiss_input.txt');
c_in=[table2array(sv_input_data(:,1:2)) num2cell(table2array(sv_input_data(:,3))) table2array(sv_input_data(:,4))];
c_in=cellfun(@string,c_in);
c_in={join(c_in,'')};d_in=cellstr(c_in{1,1});

sv_perf_other_methods = readtable('Paper_files/benchmark/predictions_other_methods/swiss_preds.txt');
c=[table2array(sv_perf_other_methods(:,1:2)) num2cell(table2array(sv_perf_other_methods(:,3))) table2array(sv_perf_other_methods(:,4))];
c=cellfun(@string,c);
c={join(c,'')};d=cellstr(c{1,1});

[Lia,Locb]=ismember(T_benchmark_sv_test.meta_merged,d_in);
length(find(Lia==1))
[Lia,Locb]=ismember(T_benchmark_sv_test.meta_merged,d);
length(find(Lia==1))
sv_perf_other_methods_ourdataset=sv_perf_other_methods(Locb,:);

NPV=zeros(11,1);Specificity=zeros(11,1);Recall=zeros(11,1);Precision=zeros(11,1);F1score=zeros(11,1);Accuracy=zeros(11,1);MCC=zeros(11,1);TP=zeros(11,1);FN=zeros(11,1);FP=zeros(11,1);TN=zeros(11,1);
Perf_table_sv_other_methods_overall=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Logit_=sv_perf_other_methods.Logit_(Locb);TP=length(find(ismember(sv_perf_Logit_,'TP'))==1);TN=length(find(ismember(sv_perf_Logit_,'TN'))==1);FN=length(find(ismember(sv_perf_Logit_,'FN'))==1);FP=length(find(ismember(sv_perf_Logit_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Logit__=sv_perf_other_methods.Logit__(Locb);TP=length(find(ismember(sv_perf_Logit__,'TP'))==1);TN=length(find(ismember(sv_perf_Logit__,'TN'))==1);FN=length(find(ismember(sv_perf_Logit__,'FN'))==1);FP=length(find(ismember(sv_perf_Logit__,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Condel_=sv_perf_other_methods.Condel_(Locb);TP=length(find(ismember(sv_perf_Condel_,'TP'))==1);TN=length(find(ismember(sv_perf_Condel_,'TN'))==1);FN=length(find(ismember(sv_perf_Condel_,'FN'))==1);FP=length(find(ismember(sv_perf_Condel_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Condel__=sv_perf_other_methods.Condel__(Locb);TP=length(find(ismember(sv_perf_Condel__,'TP'))==1);TN=length(find(ismember(sv_perf_Condel__,'TN'))==1);FN=length(find(ismember(sv_perf_Condel__,'FN'))==1);FP=length(find(ismember(sv_perf_Condel__,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_FatHMM_W_=sv_perf_other_methods.FatHMM_W_(Locb);TP=length(find(ismember(sv_perf_FatHMM_W_,'TP'))==1);TN=length(find(ismember(sv_perf_FatHMM_W_,'TN'))==1);FN=length(find(ismember(sv_perf_FatHMM_W_,'FN'))==1);FP=length(find(ismember(sv_perf_FatHMM_W_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_FatHMM_U_=sv_perf_other_methods.FatHMM_U_(Locb);TP=length(find(ismember(sv_perf_FatHMM_U_,'TP'))==1);TN=length(find(ismember(sv_perf_FatHMM_U_,'TN'))==1);FN=length(find(ismember(sv_perf_FatHMM_U_,'FN'))==1);FP=length(find(ismember(sv_perf_FatHMM_U_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(6,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_LRT_=sv_perf_other_methods.LRT_(Locb);TP=length(find(ismember(sv_perf_LRT_,'TP'))==1);TN=length(find(ismember(sv_perf_LRT_,'TN'))==1);FN=length(find(ismember(sv_perf_LRT_,'FN'))==1);FP=length(find(ismember(sv_perf_LRT_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(7,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_SIFT_=sv_perf_other_methods.SIFT_(Locb);TP=length(find(ismember(sv_perf_SIFT_,'TP'))==1);TN=length(find(ismember(sv_perf_SIFT_,'TN'))==1);FN=length(find(ismember(sv_perf_SIFT_,'FN'))==1);FP=length(find(ismember(sv_perf_SIFT_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(8,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_MutationTaster_=sv_perf_other_methods.MutationTaster_(Locb);TP=length(find(ismember(sv_perf_MutationTaster_,'TP'))==1);TN=length(find(ismember(sv_perf_MutationTaster_,'TN'))==1);FN=length(find(ismember(sv_perf_MutationTaster_,'FN'))==1);FP=length(find(ismember(sv_perf_MutationTaster_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(9,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_MutationAssessor_=sv_perf_other_methods.MutationAssessor_(Locb);TP=length(find(ismember(sv_perf_MutationAssessor_,'TP'))==1);TN=length(find(ismember(sv_perf_MutationAssessor_,'TN'))==1);FN=length(find(ismember(sv_perf_MutationAssessor_,'FN'))==1);FP=length(find(ismember(sv_perf_MutationAssessor_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(10,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_PolyPhen2_=sv_perf_other_methods.PolyPhen2_(Locb);TP=length(find(ismember(sv_perf_PolyPhen2_,'TP'))==1);TN=length(find(ismember(sv_perf_PolyPhen2_,'TN'))==1);FN=length(find(ismember(sv_perf_PolyPhen2_,'FN'))==1);FP=length(find(ismember(sv_perf_PolyPhen2_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_other_methods_overall(11,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_sv_other_methods_overall(end+1,:)=[table({'Our method'}) Perf_table_sv_test(1,2:end)];

save SwissVar_benchmark_modeling/Perf_tables_sv_other_methods.mat Perf_table_sv_other_methods_overall



% Challenging dataset construction and performance calculation:

sv_perf_Ours=cell(length(testPredictions),1);
for i=1:length(testPredictions)
    if testPredictions(i,1)==1 && neut_delet_sv_test(i,1)==1
        sv_perf_Ours(i,1)=cellstr('TP');
    end
    if testPredictions(i,1)==1 && neut_delet_sv_test(i,1)==0
        sv_perf_Ours(i,1)=cellstr('FP');
    end
    if testPredictions(i,1)==0 && neut_delet_sv_test(i,1)==1
        sv_perf_Ours(i,1)=cellstr('FN');
    end
    if testPredictions(i,1)==0 && neut_delet_sv_test(i,1)==0
        sv_perf_Ours(i,1)=cellstr('TN');
    end
end

ind_neut=find(neut_delet_sv_test==0);
ind_dele=find(neut_delet_sv_test==1);

sv_perf_Logit_neut=sv_perf_Logit_(ind_neut);
sv_perf_Logit__neut=sv_perf_Logit__(ind_neut);
sv_perf_Condel_neut=sv_perf_Condel_(ind_neut);
sv_perf_Condel__neut=sv_perf_Condel__(ind_neut);
sv_perf_FatHMM_W_neut=sv_perf_FatHMM_W_(ind_neut);
sv_perf_FatHMM_U_neut=sv_perf_FatHMM_U_(ind_neut);
sv_perf_LRT_neut=sv_perf_LRT_(ind_neut);
sv_perf_SIFT_neut=sv_perf_SIFT_(ind_neut);
sv_perf_MutationTaster_neut=sv_perf_MutationTaster_(ind_neut);
sv_perf_MutationAssessor_neut=sv_perf_MutationAssessor_(ind_neut);
sv_perf_PolyPhen2_neut=sv_perf_PolyPhen2_(ind_neut);
sv_perf_Ours_neut=sv_perf_Ours(ind_neut);

neut_T_count=zeros(length(ind_neut),1);
for i=1:length(sv_perf_Logit_neut)
    co=0;
    if ismember(sv_perf_Logit_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_Logit__neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_Condel_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_Condel__neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_FatHMM_W_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_FatHMM_U_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_LRT_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_SIFT_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_MutationTaster_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_MutationAssessor_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_PolyPhen2_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(sv_perf_Ours_neut(i,1),'TN')==1
        co=co+1;
    end
    neut_T_count(i,1)=co;
end
ind_neut_chal=ind_neut(find(neut_T_count<6));

sv_perf_Logit_dele=sv_perf_Logit_(ind_dele);
sv_perf_Logit__dele=sv_perf_Logit__(ind_dele);
sv_perf_Condel_dele=sv_perf_Condel_(ind_dele);
sv_perf_Condel__dele=sv_perf_Condel__(ind_dele);
sv_perf_FatHMM_W_dele=sv_perf_FatHMM_W_(ind_dele);
sv_perf_FatHMM_U_dele=sv_perf_FatHMM_U_(ind_dele);
sv_perf_LRT_dele=sv_perf_LRT_(ind_dele);
sv_perf_SIFT_dele=sv_perf_SIFT_(ind_dele);
sv_perf_MutationTaster_dele=sv_perf_MutationTaster_(ind_dele);
sv_perf_MutationAssessor_dele=sv_perf_MutationAssessor_(ind_dele);
sv_perf_PolyPhen2_dele=sv_perf_PolyPhen2_(ind_dele);
sv_perf_Ours_dele=sv_perf_Ours(ind_dele);

dele_T_count=zeros(length(ind_dele),1);
for i=1:length(sv_perf_Logit_dele)
    co=0;
    if ismember(sv_perf_Logit_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_Logit__dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_Condel_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_Condel__dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_FatHMM_W_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_FatHMM_U_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_LRT_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_SIFT_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_MutationTaster_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_MutationAssessor_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_PolyPhen2_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(sv_perf_Ours_dele(i,1),'TP')==1
        co=co+1;
    end
    dele_T_count(i,1)=co;
end
ind_dele_chal=ind_dele(find(dele_T_count<6));

ind_chal=unique([ind_neut_chal;ind_dele_chal]);
neut_delet_sv_test_chal=neut_delet_sv_test(ind_chal,1);

Methods_chal=({'Logit';'Logit+';'Condel';'Condel+';'FatHMM-W';'FatHMM-U';'LRT';'SIFT';'MutationTaster';'MutationAssessor';'PolyPhen2';'Our method'});
NPV=zeros(12,1);Specificity=zeros(12,1);Recall=zeros(12,1);Precision=zeros(12,1);F1score=zeros(12,1);Accuracy=zeros(12,1);MCC=zeros(12,1);TP=zeros(12,1);FN=zeros(12,1);FP=zeros(12,1);TN=zeros(12,1);
Perf_table_sv_all_methods_challenging=table(Methods_chal,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

sv_perf_Logit_chal=sv_perf_Logit_(ind_chal);TP=length(find(ismember(sv_perf_Logit_chal,'TP'))==1);TN=length(find(ismember(sv_perf_Logit_chal,'TN'))==1);FN=length(find(ismember(sv_perf_Logit_chal,'FN'))==1);FP=length(find(ismember(sv_perf_Logit_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Logit__chal=sv_perf_Logit__(ind_chal);TP=length(find(ismember(sv_perf_Logit__chal,'TP'))==1);TN=length(find(ismember(sv_perf_Logit__chal,'TN'))==1);FN=length(find(ismember(sv_perf_Logit__chal,'FN'))==1);FP=length(find(ismember(sv_perf_Logit__chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Condel_chal=sv_perf_Condel_(ind_chal);TP=length(find(ismember(sv_perf_Condel_chal,'TP'))==1);TN=length(find(ismember(sv_perf_Condel_chal,'TN'))==1);FN=length(find(ismember(sv_perf_Condel_chal,'FN'))==1);FP=length(find(ismember(sv_perf_Condel_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Condel__chal=sv_perf_Condel__(ind_chal);TP=length(find(ismember(sv_perf_Condel__chal,'TP'))==1);TN=length(find(ismember(sv_perf_Condel__chal,'TN'))==1);FN=length(find(ismember(sv_perf_Condel__chal,'FN'))==1);FP=length(find(ismember(sv_perf_Condel__chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_FatHMM_W_chal=sv_perf_FatHMM_W_(ind_chal);TP=length(find(ismember(sv_perf_FatHMM_W_chal,'TP'))==1);TN=length(find(ismember(sv_perf_FatHMM_W_chal,'TN'))==1);FN=length(find(ismember(sv_perf_FatHMM_W_chal,'FN'))==1);FP=length(find(ismember(sv_perf_FatHMM_W_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_FatHMM_U_chal=sv_perf_FatHMM_U_(ind_chal);TP=length(find(ismember(sv_perf_FatHMM_U_chal,'TP'))==1);TN=length(find(ismember(sv_perf_FatHMM_U_chal,'TN'))==1);FN=length(find(ismember(sv_perf_FatHMM_U_chal,'FN'))==1);FP=length(find(ismember(sv_perf_FatHMM_U_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(6,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_LRT_chal=sv_perf_LRT_(ind_chal);TP=length(find(ismember(sv_perf_LRT_chal,'TP'))==1);TN=length(find(ismember(sv_perf_LRT_chal,'TN'))==1);FN=length(find(ismember(sv_perf_LRT_chal,'FN'))==1);FP=length(find(ismember(sv_perf_LRT_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(7,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_SIFT_chal=sv_perf_SIFT_(ind_chal);TP=length(find(ismember(sv_perf_SIFT_chal,'TP'))==1);TN=length(find(ismember(sv_perf_SIFT_chal,'TN'))==1);FN=length(find(ismember(sv_perf_SIFT_chal,'FN'))==1);FP=length(find(ismember(sv_perf_SIFT_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(8,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_MutationTaster_chal=sv_perf_MutationTaster_(ind_chal);TP=length(find(ismember(sv_perf_MutationTaster_chal,'TP'))==1);TN=length(find(ismember(sv_perf_MutationTaster_chal,'TN'))==1);FN=length(find(ismember(sv_perf_MutationTaster_chal,'FN'))==1);FP=length(find(ismember(sv_perf_MutationTaster_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(9,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_MutationAssessor_chal=sv_perf_MutationAssessor_(ind_chal);TP=length(find(ismember(sv_perf_MutationAssessor_chal,'TP'))==1);TN=length(find(ismember(sv_perf_MutationAssessor_chal,'TN'))==1);FN=length(find(ismember(sv_perf_MutationAssessor_chal,'FN'))==1);FP=length(find(ismember(sv_perf_MutationAssessor_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(10,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_PolyPhen2_chal=sv_perf_PolyPhen2_(ind_chal);TP=length(find(ismember(sv_perf_PolyPhen2_chal,'TP'))==1);TN=length(find(ismember(sv_perf_PolyPhen2_chal,'TN'))==1);FN=length(find(ismember(sv_perf_PolyPhen2_chal,'FN'))==1);FP=length(find(ismember(sv_perf_PolyPhen2_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(11,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
sv_perf_Ours_chal=sv_perf_Ours(ind_chal);TP=length(find(ismember(sv_perf_Ours_chal,'TP'))==1);TN=length(find(ismember(sv_perf_Ours_chal,'TN'))==1);FN=length(find(ismember(sv_perf_Ours_chal,'FN'))==1);FP=length(find(ismember(sv_perf_Ours_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_sv_all_methods_challenging(12,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

save SwissVar_benchmark_modeling/Perf_tables_sv_all_methods_challenging.mat Perf_table_sv_all_methods_challenging


save SwissVar_benchmark_modeling/SwissVar_benchmark_variables.mat

