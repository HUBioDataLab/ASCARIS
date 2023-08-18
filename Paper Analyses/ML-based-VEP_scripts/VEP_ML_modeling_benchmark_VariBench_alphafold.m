% VariBench Benchmark

% Data loading:

load training_dataset_full_imputed.mat
load T_full.mat
load neut_delet.mat

opts = detectImportOptions('benchmark_varibench_training_2014selected_alphafold.txt');
opts = setvartype(opts,opts.VariableNames(1,[10:11 14 15:44]),'char');
opts = setvartype(opts,opts.VariableNames(1,[6:9 12:13 45:74]),'double');

T_benchmark_vb = readtable('benchmark_varibench_training_2014selected_alphafold.txt',opts);
[Lia,Locb]=ismember(T_final_imp.meta_merged,T_benchmark_vb.meta_merged);
T_final_imp_vb_train=T_final_imp(Lia==1,:);
T_full_vb=T_full(Lia==1,:);
neut_delet_vb=neut_delet(Lia==1,:);

T_benchmark_vb_test = readtable('benchmark_varibench_test_alphafold.txt',opts);
[Lia,Locb]=ismember(T_final_imp.meta_merged,T_benchmark_vb_test.meta_merged);
T_final_imp_vb_test=T_final_imp(Lia==1,:);
T_full_vb_test=T_full(Lia==1,:);
neut_delet_vb_test=neut_delet(Lia==1,:);
merged_vb_test=T_final_imp.meta_merged(Lia==1,:);

save VariBench_benchmark_modeling/T_all_vb.mat T_full_vb T_full_vb_test neut_delet_vb neut_delet_vb_test T_benchmark_vb T_benchmark_vb_test



% Training and testing:

Mdl_full_cval_vb=fitcensemble(T_full_vb,neut_delet_vb,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_full_cval_vb);
confmat=confusionmat(Mdl_full_cval_vb.Y,validationPredictions);
confmat=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_full_cval_vb.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_full_cval_vb.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_vb_cval=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

Mdl_full_dataset=fitcensemble(T_full_vb,neut_delet_vb,'Method','Bag');
[testPredictions,testScores]=predict(Mdl_full_test_vb,T_full_vb_test);
confmat=confusionmat(neut_delet_vb_test,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_vb_test==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_vb_test,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_vb_test=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

save VariBench_benchmark_modeling/Mdl_full_vb.mat Mdl_full_cval_vb Mdl_full_test_vb
save VariBench_benchmark_modeling/Perf_tables_vb.mat Perf_table_vb_cval Perf_table_vb_test



% Training the finalized model (Hyper-parameter optimization):

n = size(T_full_vb,1);
m = floor(log(n - 1)/log(3));
maxNumSplits = 3.^(0:m);
maxNumSplits = [maxNumSplits([8]) size(T_full_vb,1)-1];
numMNS = numel(maxNumSplits);
numTrees = [300 500];
numT = numel(numTrees);
numvartosamp = [8 24 size(T_full_vb,2)];
numV = numel(numvartosamp);
Mdl_rf_vb_hypopt = cell(numT,numMNS,numV);
AUC=zeros(numT*numV*numMNS,1);Recall=zeros(numT*numV*numMNS,1);Precision=zeros(numT*numV*numMNS,1);F1score=zeros(numT*numV*numMNS,1);Accuracy=zeros(numT*numV*numMNS,1);MCC=zeros(numT*numV*numMNS,1);TP=zeros(numT*numV*numMNS,1);FP=zeros(numT*numV*numMNS,1);FN=zeros(numT*numV*numMNS,1);TN=zeros(numT*numV*numMNS,1);
NumVartoSam=repmat(numvartosamp',numT*numMNS,1);
maxNumSplit=repmat([repmat(maxNumSplits(1,1),numV,1);repmat(maxNumSplits(1,2),numV,1)],numT,1);
numTree=repmat(numTrees,numV*numMNS,1);numTree=reshape(numTree,[size(numTree,1)*size(numTree,2),1]);
Perf_table_vb_test_hyperpar=table(numTree,maxNumSplit,NumVartoSam,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
to=0;
for i = 1:numT
    for j = 1:numMNS
        for k = 1:numV
            to=to+1;
            t = templateTree('MaxNumSplits',maxNumSplits(j),'NumVariablesToSample',numvartosamp(k));
            disp(['numTrees: ', num2str(numTrees(i)), ', maxNumSplit: ', num2str(maxNumSplits(j)), ', NumVariablesToSample: ', num2str(numvartosamp(k))])
            Mdl_rf_vb_hypopt{i,j,k} = fitcensemble(T_full_vb,neut_delet_vb,'Method','Bag','NumLearningCycles',numTrees(i),'Learners',t);
            [testPredictions,testScores]=predict(Mdl_rf_vb_hypopt{i,j,k},T_full_vb_test);
            confmat=confusionmat(neut_delet_vb_test,testPredictions);
            confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
            [~,~,~,AUC]=perfcurve(neut_delet_vb_test,testScores(:,2),1);
            TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
            Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
            Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
            Perf_table_vb_test_hyperpar(to,4:13)=Perf_table_temp;
        end
    end
end

save VariBench_benchmark_modeling/Mdl_vb_hypopt.mat Mdl_rf_vb_hypopt
save VariBench_benchmark_modeling/Perf_tables_vb_hyperpar.mat Perf_table_vb_test_hyperpar

[testPredictions,testScores]=predict(Mdl_rf_vb_hypopt{1,2,2},T_full_vb_test);
confmat=confusionmat(neut_delet_vb_test,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
[~,~,~,AUC]=perfcurve(neut_delet_vb_test,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_vb_test=table(AUC,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
save VariBench_benchmark_modeling/Perf_tables_vb.mat Perf_table_vb_cval Perf_table_vb_test



% Other methods' performance calculations:

Methods=({'Logit';'Logit+';'Condel';'Condel+';'FatHMM-W';'FatHMM-U';'LRT';'SIFT';'MutationTaster';'MutationAssessor';'PolyPhen2'});

vb_perf_other_methods = readtable('VariBench_benchmark_modeling/varibench_preds.txt');
c=[table2array(vb_perf_other_methods(:,1:2)) num2cell(table2array(vb_perf_other_methods(:,3))) table2array(vb_perf_other_methods(:,4))];
c=cellfun(@string,c);
c={join(c,'')};d=cellstr(c{1,1});

[Lia,Locb]=ismember(T_final_imp_vb_test.meta_merged,d);
length(find(Lia==1))
vb_perf_other_methods_ourdataset=vb_perf_other_methods(Locb,:);
vb_perf_other_methods_ourdataset.label(vb_perf_other_methods_ourdataset.label==-1)=0;

%(fixing the inaccurate variant labels from the respective study)
dif_ind=find(abs(vb_perf_other_methods_ourdataset.label-T_final_imp_vb_test.Var76)==1);
for i=1:length(dif_ind)
    temp=vb_perf_other_methods_ourdataset(dif_ind(i),19:30);
    temp2=table2array(vb_perf_other_methods_ourdataset(dif_ind(i),20:30));
    temp.label=abs(table2array(temp(1,1))-1);
    if temp.label==0
        temp2(1,ismember(temp2,'TP')==1)=cellstr('FP');
        temp2(1,ismember(temp2,'FN')==1)=cellstr('TN');
    else
        temp2(1,ismember(temp2,'FP')==1)=cellstr('TP');
        temp2(1,ismember(temp2,'TN')==1)=cellstr('FN');
    end
    temp(1,2:end)=temp2;
    vb_perf_other_methods_ourdataset(dif_ind(i),19:30)=temp;
end


NPV=zeros(11,1);Specificity=zeros(11,1);Recall=zeros(11,1);Precision=zeros(11,1);F1score=zeros(11,1);Accuracy=zeros(11,1);MCC=zeros(11,1);TP=zeros(11,1);FN=zeros(11,1);FP=zeros(11,1);TN=zeros(11,1);
Perf_table_vb_other_methods_overall=table(Methods,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Logit_=vb_perf_other_methods_ourdataset.Logit_;TP=length(find(ismember(vb_perf_Logit_,'TP'))==1);TN=length(find(ismember(vb_perf_Logit_,'TN'))==1);FN=length(find(ismember(vb_perf_Logit_,'FN'))==1);FP=length(find(ismember(vb_perf_Logit_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Logit__=vb_perf_other_methods_ourdataset.Logit__;TP=length(find(ismember(vb_perf_Logit__,'TP'))==1);TN=length(find(ismember(vb_perf_Logit__,'TN'))==1);FN=length(find(ismember(vb_perf_Logit__,'FN'))==1);FP=length(find(ismember(vb_perf_Logit__,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Condel_=vb_perf_other_methods_ourdataset.Condel_;TP=length(find(ismember(vb_perf_Condel_,'TP'))==1);TN=length(find(ismember(vb_perf_Condel_,'TN'))==1);FN=length(find(ismember(vb_perf_Condel_,'FN'))==1);FP=length(find(ismember(vb_perf_Condel_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Condel__=vb_perf_other_methods_ourdataset.Condel__;TP=length(find(ismember(vb_perf_Condel__,'TP'))==1);TN=length(find(ismember(vb_perf_Condel__,'TN'))==1);FN=length(find(ismember(vb_perf_Condel__,'FN'))==1);FP=length(find(ismember(vb_perf_Condel__,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_FatHMM_W_=vb_perf_other_methods_ourdataset.FatHMM_W_;TP=length(find(ismember(vb_perf_FatHMM_W_,'TP'))==1);TN=length(find(ismember(vb_perf_FatHMM_W_,'TN'))==1);FN=length(find(ismember(vb_perf_FatHMM_W_,'FN'))==1);FP=length(find(ismember(vb_perf_FatHMM_W_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_FatHMM_U_=vb_perf_other_methods_ourdataset.FatHMM_U_;TP=length(find(ismember(vb_perf_FatHMM_U_,'TP'))==1);TN=length(find(ismember(vb_perf_FatHMM_U_,'TN'))==1);FN=length(find(ismember(vb_perf_FatHMM_U_,'FN'))==1);FP=length(find(ismember(vb_perf_FatHMM_U_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(6,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_LRT_=vb_perf_other_methods_ourdataset.LRT_;TP=length(find(ismember(vb_perf_LRT_,'TP'))==1);TN=length(find(ismember(vb_perf_LRT_,'TN'))==1);FN=length(find(ismember(vb_perf_LRT_,'FN'))==1);FP=length(find(ismember(vb_perf_LRT_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(7,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_SIFT_=vb_perf_other_methods_ourdataset.SIFT_;TP=length(find(ismember(vb_perf_SIFT_,'TP'))==1);TN=length(find(ismember(vb_perf_SIFT_,'TN'))==1);FN=length(find(ismember(vb_perf_SIFT_,'FN'))==1);FP=length(find(ismember(vb_perf_SIFT_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(8,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_MutationTaster_=vb_perf_other_methods_ourdataset.MutationTaster_;TP=length(find(ismember(vb_perf_MutationTaster_,'TP'))==1);TN=length(find(ismember(vb_perf_MutationTaster_,'TN'))==1);FN=length(find(ismember(vb_perf_MutationTaster_,'FN'))==1);FP=length(find(ismember(vb_perf_MutationTaster_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(9,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_MutationAssessor_=vb_perf_other_methods_ourdataset.MutationAssessor_;TP=length(find(ismember(vb_perf_MutationAssessor_,'TP'))==1);TN=length(find(ismember(vb_perf_MutationAssessor_,'TN'))==1);FN=length(find(ismember(vb_perf_MutationAssessor_,'FN'))==1);FP=length(find(ismember(vb_perf_MutationAssessor_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(10,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_PolyPhen2_=vb_perf_other_methods_ourdataset.PolyPhen2_;TP=length(find(ismember(vb_perf_PolyPhen2_,'TP'))==1);TN=length(find(ismember(vb_perf_PolyPhen2_,'TN'))==1);FN=length(find(ismember(vb_perf_PolyPhen2_,'FN'))==1);FP=length(find(ismember(vb_perf_PolyPhen2_,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_other_methods_overall(11,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_vb_other_methods_overall(end+1,:)=[table({'Our method'}) Perf_table_vb_test(1,2:end)];

save VariBench_benchmark_modeling/Perf_tables_vb_other_methods.mat Perf_table_vb_other_methods_overall



% Challenging dataset construction and performance calculation:

vb_perf_Ours=cell(length(testPredictions),1);
for i=1:length(testPredictions)
    if testPredictions(i,1)==1 && neut_delet_vb_test(i,1)==1
        vb_perf_Ours(i,1)=cellstr('TP');
    end
    if testPredictions(i,1)==1 && neut_delet_vb_test(i,1)==0
        vb_perf_Ours(i,1)=cellstr('FP');
    end
    if testPredictions(i,1)==0 && neut_delet_vb_test(i,1)==1
        vb_perf_Ours(i,1)=cellstr('FN');
    end
    if testPredictions(i,1)==0 && neut_delet_vb_test(i,1)==0
        vb_perf_Ours(i,1)=cellstr('TN');
    end
end

ind_neut=find(neut_delet_vb_test==0);
ind_dele=find(neut_delet_vb_test==1);

vb_perf_Logit_neut=vb_perf_Logit_(ind_neut);
vb_perf_Logit__neut=vb_perf_Logit__(ind_neut);
vb_perf_Condel_neut=vb_perf_Condel_(ind_neut);
vb_perf_Condel__neut=vb_perf_Condel__(ind_neut);
vb_perf_FatHMM_W_neut=vb_perf_FatHMM_W_(ind_neut);
vb_perf_FatHMM_U_neut=vb_perf_FatHMM_U_(ind_neut);
vb_perf_LRT_neut=vb_perf_LRT_(ind_neut);
vb_perf_SIFT_neut=vb_perf_SIFT_(ind_neut);
vb_perf_MutationTaster_neut=vb_perf_MutationTaster_(ind_neut);
vb_perf_MutationAssessor_neut=vb_perf_MutationAssessor_(ind_neut);
vb_perf_PolyPhen2_neut=vb_perf_PolyPhen2_(ind_neut);
vb_perf_Ours_neut=vb_perf_Ours(ind_neut);

neut_T_count=zeros(length(ind_neut),1);
for i=1:length(vb_perf_Logit_neut)
    co=0;
    if ismember(vb_perf_Logit_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_Logit__neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_Condel_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_Condel__neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_FatHMM_W_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_FatHMM_U_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_LRT_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_SIFT_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_MutationTaster_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_MutationAssessor_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_PolyPhen2_neut(i,1),'TN')==1
        co=co+1;
    end
    if ismember(vb_perf_Ours_neut(i,1),'TN')==1
        co=co+1;
    end
    neut_T_count(i,1)=co;
end
ind_neut_chal=ind_neut(find(neut_T_count<6));

vb_perf_Logit_dele=vb_perf_Logit_(ind_dele);
vb_perf_Logit__dele=vb_perf_Logit__(ind_dele);
vb_perf_Condel_dele=vb_perf_Condel_(ind_dele);
vb_perf_Condel__dele=vb_perf_Condel__(ind_dele);
vb_perf_FatHMM_W_dele=vb_perf_FatHMM_W_(ind_dele);
vb_perf_FatHMM_U_dele=vb_perf_FatHMM_U_(ind_dele);
vb_perf_LRT_dele=vb_perf_LRT_(ind_dele);
vb_perf_SIFT_dele=vb_perf_SIFT_(ind_dele);
vb_perf_MutationTaster_dele=vb_perf_MutationTaster_(ind_dele);
vb_perf_MutationAssessor_dele=vb_perf_MutationAssessor_(ind_dele);
vb_perf_PolyPhen2_dele=vb_perf_PolyPhen2_(ind_dele);
vb_perf_Ours_dele=vb_perf_Ours(ind_dele);

dele_T_count=zeros(length(ind_dele),1);
for i=1:length(vb_perf_Logit_dele)
    co=0;
    if ismember(vb_perf_Logit_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_Logit__dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_Condel_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_Condel__dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_FatHMM_W_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_FatHMM_U_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_LRT_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_SIFT_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_MutationTaster_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_MutationAssessor_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_PolyPhen2_dele(i,1),'TP')==1
        co=co+1;
    end
    if ismember(vb_perf_Ours_dele(i,1),'TP')==1
        co=co+1;
    end
    dele_T_count(i,1)=co;
end
ind_dele_chal=ind_dele(find(dele_T_count<6));

ind_chal=unique([ind_neut_chal;ind_dele_chal]);
neut_delet_vb_test_chal=neut_delet_vb_test(ind_chal,1);

Methods_chal=({'Logit';'Logit+';'Condel';'Condel+';'FatHMM-W';'FatHMM-U';'LRT';'SIFT';'MutationTaster';'MutationAssessor';'PolyPhen2';'Our method'});
NPV=zeros(12,1);Specificity=zeros(12,1);Recall=zeros(12,1);Precision=zeros(12,1);F1score=zeros(12,1);Accuracy=zeros(12,1);MCC=zeros(12,1);TP=zeros(12,1);FN=zeros(12,1);FP=zeros(12,1);TN=zeros(12,1);
Perf_table_vb_all_methods_challenging=table(Methods_chal,NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

vb_perf_Logit_chal=vb_perf_Logit_(ind_chal);TP=length(find(ismember(vb_perf_Logit_chal,'TP'))==1);TN=length(find(ismember(vb_perf_Logit_chal,'TN'))==1);FN=length(find(ismember(vb_perf_Logit_chal,'FN'))==1);FP=length(find(ismember(vb_perf_Logit_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(1,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Logit__chal=vb_perf_Logit__(ind_chal);TP=length(find(ismember(vb_perf_Logit__chal,'TP'))==1);TN=length(find(ismember(vb_perf_Logit__chal,'TN'))==1);FN=length(find(ismember(vb_perf_Logit__chal,'FN'))==1);FP=length(find(ismember(vb_perf_Logit__chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(2,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Condel_chal=vb_perf_Condel_(ind_chal);TP=length(find(ismember(vb_perf_Condel_chal,'TP'))==1);TN=length(find(ismember(vb_perf_Condel_chal,'TN'))==1);FN=length(find(ismember(vb_perf_Condel_chal,'FN'))==1);FP=length(find(ismember(vb_perf_Condel_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(3,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Condel__chal=vb_perf_Condel__(ind_chal);TP=length(find(ismember(vb_perf_Condel__chal,'TP'))==1);TN=length(find(ismember(vb_perf_Condel__chal,'TN'))==1);FN=length(find(ismember(vb_perf_Condel__chal,'FN'))==1);FP=length(find(ismember(vb_perf_Condel__chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(4,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_FatHMM_W_chal=vb_perf_FatHMM_W_(ind_chal);TP=length(find(ismember(vb_perf_FatHMM_W_chal,'TP'))==1);TN=length(find(ismember(vb_perf_FatHMM_W_chal,'TN'))==1);FN=length(find(ismember(vb_perf_FatHMM_W_chal,'FN'))==1);FP=length(find(ismember(vb_perf_FatHMM_W_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(5,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_FatHMM_U_chal=vb_perf_FatHMM_U_(ind_chal);TP=length(find(ismember(vb_perf_FatHMM_U_chal,'TP'))==1);TN=length(find(ismember(vb_perf_FatHMM_U_chal,'TN'))==1);FN=length(find(ismember(vb_perf_FatHMM_U_chal,'FN'))==1);FP=length(find(ismember(vb_perf_FatHMM_U_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(6,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_LRT_chal=vb_perf_LRT_(ind_chal);TP=length(find(ismember(vb_perf_LRT_chal,'TP'))==1);TN=length(find(ismember(vb_perf_LRT_chal,'TN'))==1);FN=length(find(ismember(vb_perf_LRT_chal,'FN'))==1);FP=length(find(ismember(vb_perf_LRT_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(7,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_SIFT_chal=vb_perf_SIFT_(ind_chal);TP=length(find(ismember(vb_perf_SIFT_chal,'TP'))==1);TN=length(find(ismember(vb_perf_SIFT_chal,'TN'))==1);FN=length(find(ismember(vb_perf_SIFT_chal,'FN'))==1);FP=length(find(ismember(vb_perf_SIFT_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(8,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_MutationTaster_chal=vb_perf_MutationTaster_(ind_chal);TP=length(find(ismember(vb_perf_MutationTaster_chal,'TP'))==1);TN=length(find(ismember(vb_perf_MutationTaster_chal,'TN'))==1);FN=length(find(ismember(vb_perf_MutationTaster_chal,'FN'))==1);FP=length(find(ismember(vb_perf_MutationTaster_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(9,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_MutationAssessor_chal=vb_perf_MutationAssessor_(ind_chal);TP=length(find(ismember(vb_perf_MutationAssessor_chal,'TP'))==1);TN=length(find(ismember(vb_perf_MutationAssessor_chal,'TN'))==1);FN=length(find(ismember(vb_perf_MutationAssessor_chal,'FN'))==1);FP=length(find(ismember(vb_perf_MutationAssessor_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(10,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_PolyPhen2_chal=vb_perf_PolyPhen2_(ind_chal);TP=length(find(ismember(vb_perf_PolyPhen2_chal,'TP'))==1);TN=length(find(ismember(vb_perf_PolyPhen2_chal,'TN'))==1);FN=length(find(ismember(vb_perf_PolyPhen2_chal,'FN'))==1);FP=length(find(ismember(vb_perf_PolyPhen2_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(11,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
vb_perf_Ours_chal=vb_perf_Ours(ind_chal);TP=length(find(ismember(vb_perf_Ours_chal,'TP'))==1);TN=length(find(ismember(vb_perf_Ours_chal,'TN'))==1);FN=length(find(ismember(vb_perf_Ours_chal,'FN'))==1);FP=length(find(ismember(vb_perf_Ours_chal,'FP'))==1);Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));NPV=TN/(TN+FN);Specificity=TN/(TN+FP);Perf_table_vb_all_methods_challenging(12,2:12)=table(NPV,Specificity,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

save VariBench_benchmark_modeling/Perf_tables_vb_all_methods_challenging.mat Perf_table_vb_all_methods_challenging


save VariBench_benchmark_modeling/VariBench_benchmark_variables.mat

