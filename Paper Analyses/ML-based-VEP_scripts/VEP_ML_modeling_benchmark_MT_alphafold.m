% MT Benchmark

% Data loading:

load training_dataset_full_imputed.mat
load T_full.mat
load neut_delet.mat

T_MT_train_raw = readtable('benchmark_mutationtaster_training_2014selected_alphafold.txt');
[Lia,Locb]=ismember(T_final_imp.meta_merged,T_MT_train_raw.meta_merged);
T_final_imp_MT_train=T_final_imp(Lia==1,:);
T_full_MT_train=T_full(Lia==1,:);
neut_delet_mt_train=neut_delet(Lia==1,:);

T_MT_test_raw = readtable('benchmark_mutationtaster_test_alphafold.txt');
[Lia,Locb]=ismember(T_final_imp.meta_merged,T_MT_test_raw.meta_merged);
T_final_imp_MT_test=T_final_imp(Lia==1,:);
T_full_MT_test=T_full(Lia==1,:);
neut_delet_mt_test=neut_delet(Lia==1,:);
family_mt_test=T_final_imp.Var77(Lia==1,:);
merged_mt_test=T_final_imp.meta_merged(Lia==1,:);

save MT_benchmark_modeling/T_all_mt.mat T_full_MT_train neut_delet_mt_train T_full_MT_test neut_delet_mt_test family_mt_test



% Training and testing:

Mdl_full_cval_mt=fitcensemble(T_full_MT_train,neut_delet_mt_train,'Method','Bag','CrossVal','on','KFold',5,'NumLearningCycles',500);
[validationPredictions,validationScores]=kfoldPredict(Mdl_full_cval_mt);
confmat=confusionmat(Mdl_full_cval_mt.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_full_cval_mt.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_full_cval_mt.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_mt_cval=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

Mdl_full_test_mt=fitcensemble(T_full_MT_train,neut_delet_mt_train,'Method','Bag','NumLearningCycles',500);
[testPredictions,testScores]=predict(Mdl_full_test_mt,T_full_MT_test);
confmat=confusionmat(neut_delet_mt_test,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_mt_test==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_mt_test,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);
Perf_table_mt_test=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

save MT_benchmark_modeling/Mdl_full_cval_mt.mat Mdl_full_cval_mt 
save MT_benchmark_modeling/Mdl_full_test_mt.mat Mdl_full_test_mt
save MT_benchmark_modeling/Perf_tables_mt.mat Perf_table_mt_cval Perf_table_mt_test



% Training the finalized model (Hyper-parameter optimization):

n = size(T_full_MT_train,1);
m = floor(log(n - 1)/log(3));
maxNumSplits = 3.^(0:m);
maxNumSplits = [maxNumSplits([8]) size(T_full,1)-1];
numMNS = numel(maxNumSplits);
numTrees = [300 500];
numT = numel(numTrees);
numvartosamp = [8 24 size(T_full_MT_train,2)];
numV = numel(numvartosamp);
Mdl_rf_mt_hypopt = cell(numT,numMNS,numV);
AUC=zeros(numT*numV*numMNS,1);Recall=zeros(numT*numV*numMNS,1);Precision=zeros(numT*numV*numMNS,1);F1score=zeros(numT*numV*numMNS,1);Accuracy=zeros(numT*numV*numMNS,1);MCC=zeros(numT*numV*numMNS,1);TP=zeros(numT*numV*numMNS,1);FP=zeros(numT*numV*numMNS,1);FN=zeros(numT*numV*numMNS,1);TN=zeros(numT*numV*numMNS,1);
NumVartoSam=repmat(numvartosamp',numT*numMNS,1);
maxNumSplit=repmat([repmat(maxNumSplits(1,1),numV,1);repmat(maxNumSplits(1,2),numV,1)],numT,1);
numTree=repmat(numTrees,numV*numMNS,1);numTree=reshape(numTree,[size(numTree,1)*size(numTree,2),1]);
Perf_table_mt_test_hyperpar=table(numTree,maxNumSplit,NumVartoSam,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
to=0;
for i = 1:numT
    for j = 1:numMNS
        for k = 1:numV
            to=to+1;
            t = templateTree('MaxNumSplits',maxNumSplits(j),'NumVariablesToSample',numvartosamp(k));
            disp(['numTrees: ', num2str(numTrees(i)), ', maxNumSplit: ', num2str(maxNumSplits(j)), ', NumVariablesToSample: ', num2str(numvartosamp(k))])
            Mdl_rf_mt_hypopt{i,j,k} = fitcensemble(T_full_MT_train,neut_delet_mt_train,'Method','Bag','NumLearningCycles',numTrees(i),'Learners',t);
            [testPredictions,testScores]=predict(Mdl_rf_mt_hypopt{i,j,k},T_full_MT_test);
            confmat=confusionmat(neut_delet_mt_test,testPredictions);
            confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
            [~,~,~,AUC]=perfcurve(neut_delet_mt_test,testScores(:,2),1);
            TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
            Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
            Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
            Perf_table_mt_test_hyperpar(to,4:13)=Perf_table_temp;
        end
    end
end
% (the best model scored MCC: 0.7471 which is from the previous run, not in this grid search and the hyperparamter values are; 500, MaxNumSplits: default, NumVariablesToSample: default)

save MT_benchmark_modeling/Mdl_mt_hypopt.mat Mdl_rf_mt_hypopt
save MT_benchmark_modeling/Perf_tables_mt_hyperpar.mat Perf_table_mt_test_hyperpar



% Family-based performance calculation on the MT test dataset:

Family=({'Enzymes';'Membrane_receptors';'Transcription_factors';'Ion_channels';'Epigenetic_regulators';'Others';'Others_4';'Overall'});
NPV=zeros(8,1);Specificity=zeros(8,1);AUC=zeros(8,1);Recall=zeros(8,1);Precision=zeros(8,1);F1score=zeros(8,1);Accuracy=zeros(8,1);MCC=zeros(8,1);TP=zeros(8,1);FN=zeros(8,1);FP=zeros(8,1);TN=zeros(8,1);
Perf_table_mt_test_family=table(Family,NPV,Specificity,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

load MT_benchmark_modeling/Mdl_full_test_mt.mat
[testPredictions,testScores]=predict(Mdl_full_test_mt,T_full_MT_test);

ind_enzyme=find(contains(family_mt_test,'Enzymes')==1);neut_delet_mt_test_fam{1,1}=neut_delet_mt_test(ind_enzyme,1);testPred_fam{1,1}=testPredictions(ind_enzyme,1);testScor_fam{1,1}=testScores(ind_enzyme,:);
ind_membrane=find(contains(family_mt_test,'Membrane')==1);neut_delet_mt_test_fam{2,1}=neut_delet_mt_test(ind_membrane,1);testPred_fam{2,1}=testPredictions(ind_membrane,1);testScor_fam{2,1}=testScores(ind_membrane,:);
ind_tfactor=find(contains(family_mt_test,'Transcription')==1);neut_delet_mt_test_fam{3,1}=neut_delet_mt_test(ind_tfactor,1);testPred_fam{3,1}=testPredictions(ind_tfactor,1);testScor_fam{3,1}=testScores(ind_tfactor,:);
ind_ion=find(contains(family_mt_test,'Ion')==1);neut_delet_mt_test_fam{4,1}=neut_delet_mt_test(ind_ion,1);testPred_fam{4,1}=testPredictions(ind_ion,1);testScor_fam{4,1}=testScores(ind_ion,:);
ind_epigen=find(contains(family_mt_test,'Epigenetic')==1);neut_delet_mt_test_fam{5,1}=neut_delet_mt_test(ind_epigen,1);testPred_fam{5,1}=testPredictions(ind_epigen,1);testScor_fam{5,1}=testScores(ind_epigen,:);
ind_others=find(ismissing(family_mt_test)==1);neut_delet_mt_test_fam{6,1}=neut_delet_mt_test(ind_others,1);testPred_fam{6,1}=testPredictions(ind_others,1);testScor_fam{6,1}=testScores(ind_others,:);
ind_others4=unique([ind_tfactor;ind_ion;ind_epigen;ind_others]);neut_delet_mt_test_fam{7,1}=neut_delet_mt_test(ind_others4,1);testPred_fam{7,1}=testPredictions(ind_others4,1);testScor_fam{7,1}=testScores(ind_others4,:);
neut_delet_mt_test_fam{8,1}=neut_delet_mt_test;testPred_fam{8,1}=testPredictions;testScor_fam{8,1}=testScores;

for i=1:8
    confmat=confusionmat(neut_delet_mt_test_fam{i,1},testPred_fam{i,1});confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];[~,~,~,AUC]=perfcurve(neut_delet_mt_test_fam{i,1},testScor_fam{i,1}(:,2),1);TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_temp=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
    Perf_table_mt_test_family(i,2:13)=Perf_table_temp;
end

save MT_benchmark_modeling/Perf_tables_mt_family.mat Perf_table_mt_test_family



% Performance comparison against other methods:

Methods=({'PPH2_div';'PPH2_var';'MT';'SIFT';'PROVEAN'});

mt_perf_other_methods = readtable('MT_benchmark_modeling/mt_preds.txt');
c=[table2array(mt_perf_other_methods(:,1:2)) num2cell(table2array(mt_perf_other_methods(:,3))) table2array(mt_perf_other_methods(:,4))];
c=cellfun(@string,c);c={join(c,'')};d=cellstr(c{1,1});

[Lia,Locb]=ismember(T_final_imp.meta_merged,d);
mt_perf_other_methods_ourdataset=mt_perf_other_methods(Locb(Locb>0),:);

%(checking the correspondence between our variant labels and labels from the respoective study)
dif_ind=find(abs(mt_perf_other_methods_ourdataset.label-T_final_imp_MT_test.Var76)==1);
mt_perf_ourdataset_diflab=mt_perf_other_methods_ourdataset(dif_ind,1:5);
mt_perf_ourdataset_diflab.T_label=T_final_imp_MT_test.Var76(dif_ind,1);
mt_perf_ourdataset_diflab.source=T_final_imp_MT_test.Var75(dif_ind,1);
%(checking the correct outcomes from the source showed thatn the MT dataset is inaccurate in all cases, fixing them in the respective MT results file "mt_preds.txt")
%(load the fixed dataset file using the code above and repeat the last few steps)


% Performance calculations:

%overall
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_overall=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div=mt_perf_other_methods_ourdataset.PPH2_div;TP=length(find(ismember(mt_perf_PPH2div,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_overall(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var=mt_perf_other_methods_ourdataset.PPH2_var;TP=length(find(ismember(mt_perf_PPH2var,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_overall(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT=mt_perf_other_methods_ourdataset.MT;TP=length(find(ismember(mt_perf_MT,'TP'))==1);TN=length(find(ismember(mt_perf_MT,'TN'))==1);FN=length(find(ismember(mt_perf_MT,'FN'))==1);FP=length(find(ismember(mt_perf_MT,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_overall(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT=mt_perf_other_methods_ourdataset.SIFT;TP=length(find(ismember(mt_perf_SIFT,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_overall(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN=mt_perf_other_methods_ourdataset.PROVEAN;TP=length(find(ismember(mt_perf_PROVEAN,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_overall(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_overall(end+1,:)=[table({'Our method'}) Perf_table_mt_test(1,2:end)];
%enzyme
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_enzyme=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_enzyme=mt_perf_PPH2div(ind_enzyme);TP=length(find(ismember(mt_perf_PPH2div_enzyme,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_enzyme,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_enzyme,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_enzyme,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_enzyme(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_enzyme=mt_perf_PPH2var(ind_enzyme);TP=length(find(ismember(mt_perf_PPH2var_enzyme,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_enzyme,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_enzyme,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_enzyme,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_enzyme(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_enzyme=mt_perf_MT(ind_enzyme);TP=length(find(ismember(mt_perf_MT_enzyme,'TP'))==1);TN=length(find(ismember(mt_perf_MT_enzyme,'TN'))==1);FN=length(find(ismember(mt_perf_MT_enzyme,'FN'))==1);FP=length(find(ismember(mt_perf_MT_enzyme,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_enzyme(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_enzyme=mt_perf_SIFT(ind_enzyme);TP=length(find(ismember(mt_perf_SIFT_enzyme,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_enzyme,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_enzyme,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_enzyme,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_enzyme(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_enzyme=mt_perf_PROVEAN(ind_enzyme);TP=length(find(ismember(mt_perf_PROVEAN_enzyme,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_enzyme,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_enzyme,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_enzyme,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_enzyme(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_enzyme(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(1,3:end)];
%membrane
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_membrane=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_membrane=mt_perf_PPH2div(ind_membrane);TP=length(find(ismember(mt_perf_PPH2div_membrane,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_membrane,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_membrane,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_membrane,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_membrane(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_membrane=mt_perf_PPH2var(ind_membrane);TP=length(find(ismember(mt_perf_PPH2var_membrane,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_membrane,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_membrane,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_membrane,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_membrane(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_membrane=mt_perf_MT(ind_membrane);TP=length(find(ismember(mt_perf_MT_membrane,'TP'))==1);TN=length(find(ismember(mt_perf_MT_membrane,'TN'))==1);FN=length(find(ismember(mt_perf_MT_membrane,'FN'))==1);FP=length(find(ismember(mt_perf_MT_membrane,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_membrane(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_membrane=mt_perf_SIFT(ind_membrane);TP=length(find(ismember(mt_perf_SIFT_membrane,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_membrane,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_membrane,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_membrane,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_membrane(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_membrane=mt_perf_PROVEAN(ind_membrane);TP=length(find(ismember(mt_perf_PROVEAN_membrane,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_membrane,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_membrane,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_membrane,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_membrane(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_membrane(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(2,3:end)];
%tfactor
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_tfactor=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_tfactor=mt_perf_PPH2div(ind_tfactor);TP=length(find(ismember(mt_perf_PPH2div_tfactor,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_tfactor,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_tfactor,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_tfactor,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_tfactor(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_tfactor=mt_perf_PPH2var(ind_tfactor);TP=length(find(ismember(mt_perf_PPH2var_tfactor,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_tfactor,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_tfactor,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_tfactor,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_tfactor(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_tfactor=mt_perf_MT(ind_tfactor);TP=length(find(ismember(mt_perf_MT_tfactor,'TP'))==1);TN=length(find(ismember(mt_perf_MT_tfactor,'TN'))==1);FN=length(find(ismember(mt_perf_MT_tfactor,'FN'))==1);FP=length(find(ismember(mt_perf_MT_tfactor,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_tfactor(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_tfactor=mt_perf_SIFT(ind_tfactor);TP=length(find(ismember(mt_perf_SIFT_tfactor,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_tfactor,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_tfactor,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_tfactor,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_tfactor(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_tfactor=mt_perf_PROVEAN(ind_tfactor);TP=length(find(ismember(mt_perf_PROVEAN_tfactor,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_tfactor,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_tfactor,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_tfactor,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_tfactor(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_tfactor(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(3,3:end)];
%ion
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_ion=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_ion=mt_perf_PPH2div(ind_ion);TP=length(find(ismember(mt_perf_PPH2div_ion,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_ion,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_ion,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_ion,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_ion(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_ion=mt_perf_PPH2var(ind_ion);TP=length(find(ismember(mt_perf_PPH2var_ion,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_ion,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_ion,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_ion,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_ion(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_ion=mt_perf_MT(ind_ion);TP=length(find(ismember(mt_perf_MT_ion,'TP'))==1);TN=length(find(ismember(mt_perf_MT_ion,'TN'))==1);FN=length(find(ismember(mt_perf_MT_ion,'FN'))==1);FP=length(find(ismember(mt_perf_MT_ion,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_ion(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_ion=mt_perf_SIFT(ind_ion);TP=length(find(ismember(mt_perf_SIFT_ion,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_ion,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_ion,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_ion,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_ion(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_ion=mt_perf_PROVEAN(ind_ion);TP=length(find(ismember(mt_perf_PROVEAN_ion,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_ion,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_ion,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_ion,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_ion(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_ion(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(4,3:end)];
%epigen
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_epigen=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_epigen=mt_perf_PPH2div(ind_epigen);TP=length(find(ismember(mt_perf_PPH2div_epigen,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_epigen,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_epigen,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_epigen,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_epigen(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_epigen=mt_perf_PPH2var(ind_epigen);TP=length(find(ismember(mt_perf_PPH2var_epigen,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_epigen,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_epigen,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_epigen,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_epigen(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_epigen=mt_perf_MT(ind_epigen);TP=length(find(ismember(mt_perf_MT_epigen,'TP'))==1);TN=length(find(ismember(mt_perf_MT_epigen,'TN'))==1);FN=length(find(ismember(mt_perf_MT_epigen,'FN'))==1);FP=length(find(ismember(mt_perf_MT_epigen,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_epigen(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_epigen=mt_perf_SIFT(ind_epigen);TP=length(find(ismember(mt_perf_SIFT_epigen,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_epigen,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_epigen,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_epigen,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_epigen(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_epigen=mt_perf_PROVEAN(ind_epigen);TP=length(find(ismember(mt_perf_PROVEAN_epigen,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_epigen,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_epigen,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_epigen,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_epigen(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_epigen(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(5,3:end)];
%others
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_others=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_others=mt_perf_PPH2div(ind_others);TP=length(find(ismember(mt_perf_PPH2div_others,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_others,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_others,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_others,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_others=mt_perf_PPH2var(ind_others);TP=length(find(ismember(mt_perf_PPH2var_others,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_others,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_others,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_others,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_others=mt_perf_MT(ind_others);TP=length(find(ismember(mt_perf_MT_others,'TP'))==1);TN=length(find(ismember(mt_perf_MT_others,'TN'))==1);FN=length(find(ismember(mt_perf_MT_others,'FN'))==1);FP=length(find(ismember(mt_perf_MT_others,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_others=mt_perf_SIFT(ind_others);TP=length(find(ismember(mt_perf_SIFT_others,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_others,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_others,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_others,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_others=mt_perf_PROVEAN(ind_others);TP=length(find(ismember(mt_perf_PROVEAN_others,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_others,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_others,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_others,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_others(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(6,3:end)];
%others4
NPV=zeros(5,1);Specificity=zeros(5,1);Recall=zeros(5,1);Precision=zeros(5,1);F1score=zeros(5,1);Accuracy=zeros(5,1);MCC=zeros(5,1);TP=zeros(5,1);FN=zeros(5,1);FP=zeros(5,1);TN=zeros(5,1);
Perf_table_mt_other_methods_others4=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2div_others4=mt_perf_PPH2div(ind_others4);TP=length(find(ismember(mt_perf_PPH2div_others4,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_others4,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_others4,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_others4,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others4(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_others4=mt_perf_PPH2var(ind_others4);TP=length(find(ismember(mt_perf_PPH2var_others4,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_others4,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_others4,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_others4,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others4(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_others4=mt_perf_MT(ind_others4);TP=length(find(ismember(mt_perf_MT_others4,'TP'))==1);TN=length(find(ismember(mt_perf_MT_others4,'TN'))==1);FN=length(find(ismember(mt_perf_MT_others4,'FN'))==1);FP=length(find(ismember(mt_perf_MT_others4,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others4(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_others4=mt_perf_SIFT(ind_others4);TP=length(find(ismember(mt_perf_SIFT_others4,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_others4,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_others4,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_others4,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others4(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_others4=mt_perf_PROVEAN(ind_others4);TP=length(find(ismember(mt_perf_PROVEAN_others4,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_others4,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_others4,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_others4,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_other_methods_others4(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_mt_other_methods_others4(end+1,:)=[table({'Our method'}) Perf_table_mt_test_family(7,3:end)];

save MT_benchmark_modeling/Perf_tables_mt_other_methods.mat Perf_table_mt_other_methods_overall Perf_table_mt_other_methods_enzyme Perf_table_mt_other_methods_membrane Perf_table_mt_other_methods_tfactor Perf_table_mt_other_methods_ion Perf_table_mt_other_methods_epigen Perf_table_mt_other_methods_others Perf_table_mt_other_methods_others4



% Challenging dataset construction and performance calculation:

mt_perf_Ours=cell(length(testPredictions),1);
for i=1:length(testPredictions)
    if testPredictions(i,1)==1 && neut_delet_mt_test(i,1)==1
        mt_perf_Ours(i,1)=cellstr('TP');
    end
    if testPredictions(i,1)==1 && neut_delet_mt_test(i,1)==0
        mt_perf_Ours(i,1)=cellstr('FP');
    end
    if testPredictions(i,1)==0 && neut_delet_mt_test(i,1)==1
        mt_perf_Ours(i,1)=cellstr('FN');
    end
    if testPredictions(i,1)==0 && neut_delet_mt_test(i,1)==0
        mt_perf_Ours(i,1)=cellstr('TN');
    end
end

ind_neut=find(neut_delet_mt_test==0);
ind_dele=find(neut_delet_mt_test==1);

mt_perf_PPH2div_neut=mt_perf_PPH2div(ind_neut);
mt_perf_PPH2var_neut=mt_perf_PPH2var(ind_neut);
mt_perf_MT_neut=mt_perf_MT(ind_neut);
mt_perf_SIFT_neut=mt_perf_SIFT(ind_neut);
mt_perf_PROVEAN_neut=mt_perf_PROVEAN(ind_neut);
mt_perf_Ours_neut=mt_perf_Ours(ind_neut);

neut_T_count=zeros(length(ind_neut),1);
for i=1:length(mt_perf_PPH2div_neut)
    co=0;
    if ismember(mt_perf_PPH2div_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(mt_perf_PPH2var_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(mt_perf_MT_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(mt_perf_SIFT_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(mt_perf_PROVEAN_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(mt_perf_Ours_neut(i,1),'TN')==1
        co=co+1;
    end
    neut_T_count(i,1)=co;
end

mt_perf_PPH2div_dele=mt_perf_PPH2div(ind_dele);
mt_perf_PPH2var_dele=mt_perf_PPH2var(ind_dele);
mt_perf_MT_dele=mt_perf_MT(ind_dele);
mt_perf_SIFT_dele=mt_perf_SIFT(ind_dele);
mt_perf_PROVEAN_dele=mt_perf_PROVEAN(ind_dele);
mt_perf_Ours_dele=mt_perf_Ours(ind_dele);

dele_T_count=zeros(length(ind_dele),1);
for i=1:length(mt_perf_PPH2div_dele)
    co=0;
    if ismember(mt_perf_PPH2div_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(mt_perf_PPH2var_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(mt_perf_MT_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(mt_perf_SIFT_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(mt_perf_PROVEAN_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(mt_perf_Ours_dele(i,1),'TP')==1
        co=co+1;
    end
    dele_T_count(i,1)=co;
end

ind_neut_chal=ind_neut(neut_T_count<4);
ind_dele_chal=ind_dele(dele_T_count<4);
ind_chal=unique([ind_neut_chal;ind_dele_chal]);
neut_delet_mt_test_chal=neut_delet_mt_test(ind_chal,1);

chal_savs=T_final_imp_MT_test(ind_chal,[1 2 4 3 76]);
save MT_benchmark_modeling/chal_savs.mat chal_savs
writetable(chal_savs,'MT_benchmark_modeling/challenging_MT_test_SAVs.txt')

Methods_chal=({'PPH2_div';'PPH2_var';'MT';'SIFT';'PROVEAN';'Ours'});
Recall=zeros(6,1);Precision=zeros(6,1);F1score=zeros(6,1);Accuracy=zeros(6,1);NPV=zeros(6,1);Specificity=zeros(6,1);MCC=zeros(6,1);TP=zeros(6,1);FN=zeros(6,1);FP=zeros(6,1);TN=zeros(6,1);
Perf_table_mt_all_methods_challenging=table(Methods_chal,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

mt_perf_PPH2div_chal=mt_perf_PPH2div(ind_chal);TP=length(find(ismember(mt_perf_PPH2div_chal,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_chal,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_chal,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_all_methods_challenging(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PPH2var_chal=mt_perf_PPH2var(ind_chal);TP=length(find(ismember(mt_perf_PPH2var_chal,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_chal,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_chal,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_all_methods_challenging(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_MT_chal=mt_perf_MT(ind_chal);TP=length(find(ismember(mt_perf_MT_chal,'TP'))==1);TN=length(find(ismember(mt_perf_MT_chal,'TN'))==1);FN=length(find(ismember(mt_perf_MT_chal,'FN'))==1);FP=length(find(ismember(mt_perf_MT_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_all_methods_challenging(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_SIFT_chal=mt_perf_SIFT(ind_chal);TP=length(find(ismember(mt_perf_SIFT_chal,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_chal,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_chal,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_all_methods_challenging(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_PROVEAN_chal=mt_perf_PROVEAN(ind_chal);TP=length(find(ismember(mt_perf_PROVEAN_chal,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_chal,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_chal,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_all_methods_challenging(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
mt_perf_Ours_chal=mt_perf_Ours(ind_chal);TP=length(find(ismember(mt_perf_Ours_chal,'TP'))==1);TN=length(find(ismember(mt_perf_Ours_chal,'TN'))==1);FN=length(find(ismember(mt_perf_Ours_chal,'FN'))==1);FP=length(find(ismember(mt_perf_Ours_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_all_methods_challenging(6,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

save MT_benchmark_modeling/Perf_tables_mt_all_methods_challenging.mat Perf_table_mt_all_methods_challenging

%(normalizing the scores on the challenging dataset with respect to a random predictor)
TP=0;TN=0;FP=0;FN=0;
for j=1:1000
    randPredictions=randi([0 1], length(neut_delet_mt_test),1);
    for i=1:length(randPredictions)
        if randPredictions(i,1)==1 && neut_delet_mt_test(i,1)==1
            TP=TP+1;
        end
        if randPredictions(i,1)==1 && neut_delet_mt_test(i,1)==0
            FP=FP+1;
        end
        if randPredictions(i,1)==0 && neut_delet_mt_test(i,1)==1
            FN=FN+1;
        end
        if randPredictions(i,1)==0 && neut_delet_mt_test(i,1)==0
            TN=TN+1;
        end
    end
end
Perf_table_mt_random_predictor=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_mt_random_predictor(1,1:11)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

All=zeros(6,1);less6=zeros(6,1);less5=zeros(6,1);less4=zeros(6,1);less3=zeros(6,1);less2=zeros(6,1);
Perf_table_mt_all_methods_challenging_longtest=table(Methods_chal,All,less6,less5,less4,less3,less2);
for i=0:5
    x=7-i;
    ind_neut_chal=ind_neut(neut_T_count<x);
    ind_dele_chal=ind_dele(dele_T_count<x);
    ind_chal=unique([ind_neut_chal;ind_dele_chal]);

    mt_perf_PPH2div_chal=mt_perf_PPH2div(ind_chal);TP=length(find(ismember(mt_perf_PPH2div_chal,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2div_chal,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2div_chal,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2div_chal,'FP'))==1);Perf_table_mt_all_methods_challenging_longtest(1,i+2)=num2cell((2*TP)/(2*TP+FP+FN));
    mt_perf_PPH2var_chal=mt_perf_PPH2var(ind_chal);TP=length(find(ismember(mt_perf_PPH2var_chal,'TP'))==1);TN=length(find(ismember(mt_perf_PPH2var_chal,'TN'))==1);FN=length(find(ismember(mt_perf_PPH2var_chal,'FN'))==1);FP=length(find(ismember(mt_perf_PPH2var_chal,'FP'))==1);Perf_table_mt_all_methods_challenging_longtest(2,i+2)=num2cell((2*TP)/(2*TP+FP+FN));
    mt_perf_MT_chal=mt_perf_MT(ind_chal);TP=length(find(ismember(mt_perf_MT_chal,'TP'))==1);TN=length(find(ismember(mt_perf_MT_chal,'TN'))==1);FN=length(find(ismember(mt_perf_MT_chal,'FN'))==1);FP=length(find(ismember(mt_perf_MT_chal,'FP'))==1);Perf_table_mt_all_methods_challenging_longtest(3,i+2)=num2cell((2*TP)/(2*TP+FP+FN));
    mt_perf_SIFT_chal=mt_perf_SIFT(ind_chal);TP=length(find(ismember(mt_perf_SIFT_chal,'TP'))==1);TN=length(find(ismember(mt_perf_SIFT_chal,'TN'))==1);FN=length(find(ismember(mt_perf_SIFT_chal,'FN'))==1);FP=length(find(ismember(mt_perf_SIFT_chal,'FP'))==1);Perf_table_mt_all_methods_challenging_longtest(4,i+2)=num2cell((2*TP)/(2*TP+FP+FN));
    mt_perf_PROVEAN_chal=mt_perf_PROVEAN(ind_chal);TP=length(find(ismember(mt_perf_PROVEAN_chal,'TP'))==1);TN=length(find(ismember(mt_perf_PROVEAN_chal,'TN'))==1);FN=length(find(ismember(mt_perf_PROVEAN_chal,'FN'))==1);FP=length(find(ismember(mt_perf_PROVEAN_chal,'FP'))==1);Perf_table_mt_all_methods_challenging_longtest(5,i+2)=num2cell((2*TP)/(2*TP+FP+FN));
    mt_perf_Ours_chal=mt_perf_Ours(ind_chal);TP=length(find(ismember(mt_perf_Ours_chal,'TP'))==1);TN=length(find(ismember(mt_perf_Ours_chal,'TN'))==1);FN=length(find(ismember(mt_perf_Ours_chal,'FN'))==1);FP=length(find(ismember(mt_perf_Ours_chal,'FP'))==1);Perf_table_mt_all_methods_challenging_longtest(6,i+2)=num2cell((2*TP)/(2*TP+FP+FN));
end

figure;plot(table2array(Perf_table_mt_all_methods_challenging_longtest(:,2:end))')

save MT_benchmark_modeling/Perf_table_mt_all_methods_challenging_longtest.mat Perf_table_mt_all_methods_challenging_longtest



% Use-case example selection:

% only ours is correct:
conf_all=[mt_perf_PPH2div mt_perf_PPH2var mt_perf_MT mt_perf_SIFT mt_perf_PROVEAN mt_perf_Ours];
idx_neut = find(cellfun(@(c)all(strcmp({'FP','FP','FP','FP','FP','TN'},c)),num2cell(conf_all,2))==1);
idx_dele = find(cellfun(@(c)all(strcmp({'FN','FN','FN','FN','FN','TP'},c)),num2cell(conf_all,2))==1);
usecase_cand_neut=T_benchmark_mt_test(idx_neut,1:4);
usecase_cand_dele=T_benchmark_mt_test(idx_dele,1:4);

% only ours and 1 additional method is correct (only done for dele SAVs):
mt_perf_Ours_dele_chal=mt_perf_Ours(ind_dele_chal,1);
Lia=ismember(mt_perf_Ours_dele_chal,'TP');
idx_dele1=ind_dele_chal(Lia==1);
usecase_cand_dele1=T_benchmark_mt_test(idx_dele1,1:4);

save MT_benchmark_modeling/MT_benchmark_variables.mat


