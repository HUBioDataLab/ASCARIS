% Data load and pre-processing procedures ("Paper_files_old" folder contain unimputed original tables, "Paper_files" contain imputed tables according to the resuts of the analyses provided below):

opts = detectImportOptions('Paper_files_old/originaldata/training_uptodate_full.txt');
opts = setvartype(opts,opts.VariableNames(1,[9:10 13 44:73]),'char');
opts = setvartype(opts,opts.VariableNames(1,[5:8 11:12 14:43 77]),'double');
T_final = readtable('Paper_files_old/originaldata/training_uptodate_full.txt',opts);
T_title = T_final.Properties.VariableNames;
neut_delet = table2array(T_final(:,77));

%Feature imputation and scaling for the missing values:
T_domain_arr1=table2array(T_final(:,10));
length(find(cell2mat(cellfun(@(x)any(isnan(x)),T_domain_arr1,'UniformOutput',false))))
length(find(cell2mat(cellfun(@(x)any(isempty(x)),T_domain_arr1,'UniformOutput',false))))

T_domain_arr2=table2array(T_final(:,11));
length(find(isnan(T_domain_arr2)==1))
T_domain_arr2_imp=T_domain_arr2;
T_domain_arr2_imp(isnan(T_domain_arr2)==1)=mean(T_domain_arr2(~isnan(T_domain_arr2) & T_domain_arr2~=0));

T_physico_arr=table2array(T_final(:,[5 6 7 8]));
length(find(isnan(T_physico_arr)==1))

T_location_arr1=table2array(T_final(:,12));
length(find(isnan(T_location_arr1)==1))
T_location_arr1_imp=T_location_arr1;
T_location_arr1_imp(isnan(T_location_arr1)==1)=mean(T_location_arr1(~isnan(T_location_arr1)));

T_location_arr2=table2array(T_final(:,13));
length(find(cell2mat(cellfun(@(x)any(isnan(x)),T_location_arr2,'UniformOutput',false))))
length(find(cell2mat(cellfun(@(x)any(isempty(x)),T_location_arr2,'UniformOutput',false))))
T_location_arr2_imp=T_location_arr2;
T_location_arr2_imp(cell2mat(cellfun(@(x)any(isempty(x)),T_location_arr2,'UniformOutput',false)))=cellstr('unknown');

T_annobin_arr=table2array(T_final(:,44:73));
length(find(cell2mat(cellfun(@(x)any(isnan(x)),T_annobin_arr,'UniformOutput',false))))
length(find(cell2mat(cellfun(@(x)any(isempty(x)),T_annobin_arr,'UniformOutput',false))))

T_annodis_arr=table2array(T_final(:,14:43));
length(find(isnan(T_annodis_arr)==1))

T_annodis_arr_imp=ones(length((table2array(T_final(:,1)))),30);
for i=1:30
    T_annodis_temp_arr=table2array(T_final(:,i+13));
    if isa(T_annodis_temp_arr,'double')==0
        T_annodis_temp_arr(cell2mat(cellfun(@(x)any(isempty(x)),T_annodis_temp_arr,'UniformOutput',false)))=cellstr('NaN');
        T_annodis_temp_arr=cellfun(@str2num, T_annodis_temp_arr);
    end
    T_annodis_arr_imp(:,i)=T_annodis_temp_arr;
    T_annodis_arr_imp(isnan(T_annodis_temp_arr),i)=impute_annot_mean_all(i,3);
end

save training_dataset_orig_imputed_var.mat T_domain_arr1 T_domain_arr2 T_domain_arr2_imp T_location_arr1 T_location_arr2 T_location_arr2 T_location_arr2_imp T_annobin_arr T_annodis_arr T_annodis_arr_imp neut_delet

T_final_imp=T_final;
T_final_imp(:,11)=array2table(T_domain_arr2_imp);
T_final_imp(:,12)=array2table(T_location_arr1_imp);
T_final_imp(:,13)=array2table(T_location_arr2_imp);
T_final_imp(:,14:43)=array2table(T_annodis_arr_imp);
save training_dataset_full_imputed.mat T_final_imp neut_delet

%Generation of feature vectors:
T_domain=T_final_imp(:,10);
T_physico=T_final_imp(:,[5 6 7 8]);
T_location=T_final_imp(:,[12 13]);
T_structural=T_final_imp(:,[5 6 7 8 10 11 12 13]);
T_annobin=T_final_imp(:,44:73);
T_annobindis=T_final_imp(:,[44:73 14:43]);
T_strucannobin=T_final_imp(:,[5 6 7 8 10 11 12 13 44:73]);
T_full=T_final_imp(:,[5 6 7 8 10 11 12 13 44:73 14:43]);



% Prototype model design and test:

%Training and cross-validation:
Mdl_structural=fitcensemble(T_structural,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
%Loss calculation and visualization:
% kflc = kfoldLoss(Mdl_structural,'Mode','cumulative');
% figure;
% plot(kflc);
% ylabel('10-fold Misclassification rate');
% xlabel('Learning cycle');
%Calculation of cross-validation performance metrics:
[validationPredictions,validationScores]=kfoldPredict(Mdl_structural);
confmat=confusionmat(Mdl_structural.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_structural.Y==1 & validationPredictions==1))
% figure;
% confusionchart(Mdl_structural.Y,validationPredictions)
[~,~,~,AUC]=perfcurve(Mdl_structural.Y,validationScores(:,2),1);
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification')
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(4,2:11)=Perf_table_temp;
%Predictor Importance Estimation
t=templateTree('NumVariablesToSample','all');
Mdl_structural_all=fitcensemble(T_structural,neut_delet,'Method','Bag','Learners',t);
impOOB_structural = oobPermutedPredictorImportance(Mdl_structural_all,'Options',statset('UseParallel',true));
figure
bar(impOOB_structural)
title('Structural Features - Predictor Importance Estimates')
xlabel('Predictor variable')
ylabel('Importance')
h = gca;
h.XTickLabel = Mdl_structural_all.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';



% Featurization test (Running this pipeline for different combination of featurizations):

Feature=({'Domain';'Physico';'Location';'Structural';'Anno_Bin';'Anno_BinDis';'Struc_Anno_Bin';'Full'});
AUC=zeros(8,1);Recall=zeros(8,1);Precision=zeros(8,1);F1score=zeros(8,1);Accuracy=zeros(8,1);MCC=zeros(8,1);TP=zeros(8,1);FN=zeros(8,1);FP=zeros(8,1);TN=zeros(8,1);
Perf_table_ablation=table(Feature,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

Mdl_domain=fitcensemble(T_domain,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_domain);
confmat=confusionmat(Mdl_domain.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_domain.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_domain.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(1,2:11)=Perf_table_temp;

Mdl_physico=fitcensemble(T_physico,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_physico);
confmat=confusionmat(Mdl_physico.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_physico.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_physico.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(2,2:11)=Perf_table_temp;

Mdl_location=fitcensemble(T_location,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_location);
confmat=confusionmat(Mdl_location.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_location.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_location.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(3,2:11)=Perf_table_temp;

Mdl_structural=fitcensemble(T_structural,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_structural);
confmat=confusionmat(Mdl_structural.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_structural.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_structural.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(4,2:11)=Perf_table_temp;

Mdl_annobin=fitcensemble(T_annobin,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_annobin);
confmat=confusionmat(Mdl_annobin.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_annobin.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_annobin.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(5,2:11)=Perf_table_temp;

Mdl_annobindis=fitcensemble(T_annobindis,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_annobindis);
confmat=confusionmat(Mdl_annobindis.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_annobindis.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_annobindis.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(6,2:11)=Perf_table_temp;

Mdl_strucannobin=fitcensemble(T_strucannobin,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_strucannobin);
confmat=confusionmat(Mdl_strucannobin.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_strucannobin.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_strucannobin.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(7,2:11)=Perf_table_temp;

Mdl_full=fitcensemble(T_full,neut_delet,'Method','Bag','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_full);
confmat=confusionmat(Mdl_full.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_full.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_full.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_ablation(8,2:11)=Perf_table_temp;

t=templateTree('NumVariablesToSample','all');
Mdl_full_all=fitcensemble(T_full,neut_delet,'Method','Bag','Learners',t);
impOOB_full = oobPermutedPredictorImportance(Mdl_full_all,'Options',statset('UseParallel',true));
figure
bar(impOOB_full)
title('All Features - Predictor Importance Estimates')
xlabel('Predictor variable')
ylabel('Importance')
set(gca, 'XTick', linspace(0.75,100), 'XTickLabels', Mdl_full_all.PredictorNames, 'XTickLabelRotation', 45, 'TickLabelInterpreter', 'none')
axis([0 67.9 0 8])

save Perf_table_ablation.mat Perf_table_ablation



% Algorithm test (Experimenting with other classifiers):

Classifiers=({'RF_500';'AdaBoost';'LogitBoost';'SVM';'NBayes';'FFNN'});
AUC=zeros(6,1);Recall=zeros(6,1);Precision=zeros(6,1);F1score=zeros(6,1);Accuracy=zeros(6,1);MCC=zeros(6,1);TP=zeros(6,1);FN=zeros(6,1);FP=zeros(6,1);TN=zeros(6,1);
Perf_table_classifiers=table(Classifiers,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

Mdl_full_500=fitcensemble(T_full,neut_delet,'Method','Bag','CrossVal','on','KFold',5,'NumLearningCycles',500);
[validationPredictions,validationScores]=kfoldPredict(Mdl_full_500);
confmat=confusionmat(Mdl_full_500.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_full_500.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_full_500.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp_rf_full_500=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_classifiers(1,2:11)=Perf_table_temp_rf_full_500;

Mdl_adaboost_full=fitcensemble(T_full,neut_delet,'Method','AdaBoostM1','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_adaboost_full);
confmat=confusionmat(Mdl_adaboost_full.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_adaboost_full.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_adaboost_full.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp_adaboost_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_classifiers(2,2:11)=Perf_table_temp_adaboost_full;

Mdl_logitboost_full=fitcensemble(T_full,neut_delet,'Method','LogitBoost','CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_logitboost_full);
confmat=confusionmat(Mdl_logitboost_full.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_logitboost_full.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_logitboost_full.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp_logitboost_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_classifiers(3,2:11)=Perf_table_temp_logitboost_full;

Mdl_svm_full=fitcsvm(T_full,neut_delet,'KernelFunction','linear','CrossVal','on','KFold',5,'Verbose',1,'NumPrint',500,'IterationLimit',100000);
[validationPredictions,validationScores]=kfoldPredict(Mdl_svm_full);
confmat=confusionmat(Mdl_svm_full.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_svm_full.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_svm_full.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp_svm_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_classifiers(4,2:11)=Perf_table_temp_svm_full;

Mdl_nbayes_full=fitcnb(T_full,neut_delet,'CrossVal','on','KFold',5);
[validationPredictions,validationScores]=kfoldPredict(Mdl_nbayes_full);
confmat=confusionmat(Mdl_nbayes_full.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_nbayes_full.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_nbayes_full.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp_nbayes_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_classifiers(5,2:11)=Perf_table_temp_nbayes_full;

Mdl_ffnn_full=fitcnet(T_full,neut_delet,'CrossVal','on','KFold',5,'verbose',1,'LayerSizes',[128 64 32],'Activations','sigmoid','Standardize',true);
[validationPredictions,validationScores]=kfoldPredict(Mdl_ffnn_full);
confmat=confusionmat(Mdl_ffnn_full.Y,validationPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(Mdl_ffnn_full.Y==1 & validationPredictions==1))
[~,~,~,AUC]=perfcurve(Mdl_ffnn_full.Y,validationScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp_ffnn_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_classifiers(6,2:11)=Perf_table_temp_ffnn_full;

% LR requires numerical variables
% Mdl_lr_full=fitclinear(T_final,neut_delet_cat,'Learner','logistic','CategoricalPredictors','all','CrossVal','on','KFold',5);
% [validationPredictions,validationScores]=kfoldPredict(Mdl_lr_full);
% confmat=confusionmat(Mdl_lr_full.Y,validationPredictions);
% confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
% confmat_fix(1,1)
% length(find(Mdl_lr_full.Y==1 & validationPredictions==1))
% [~,~,~,AUC]=perfcurve(Mdl_lr_full.Y,validationScores(:,2),1);
% TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
% Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
% Perf_table_temp_svm_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
% Perf_table_classifiers(6,2:11)=Perf_table_temp_svm_full;

% KNN does not work with a mixture of categorical and numerical variables
% Mdl_knn_full=fitcknn(T_full,neut_delet,'CrossVal','off','NumNeighbors',7,'Standardize',1);
% Mdl_knn_full_cros=crossval(Mdl_knn_full,'KFold',5);
% [validationPredictions,validationScores]=kfoldPredict(Mdl_knn_full);
% confmat=confusionmat(Mdl_knn_full.Y,validationPredictions);
% confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
% confmat_fix(1,1)
% length(find(Mdl_knn_full.Y==1 & validationPredictions==1))
% [~,~,~,AUC]=perfcurve(Mdl_knn_full.Y,validationScores(:,2),1);
% TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
% Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
% Perf_table_temp_knn_full=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
% Perf_table_classifiers(7,2:11)=Perf_table_temp_knn_full;

save Perf_table_classifiers.mat Perf_table_classifiers



% Dataset / data resource test (Calculation of hold-out test performance metrics using models trained with datasets of different resources):

Resources=({'UniProt';'ClinVar';'PMD';'Full dataset'});
AUC=zeros(4,1);Recall=zeros(4,1);Precision=zeros(4,1);F1score=zeros(4,1);Accuracy=zeros(4,1);MCC=zeros(4,1);TP=zeros(4,1);FN=zeros(4,1);FP=zeros(4,1);TN=zeros(4,1);
Perf_table_dataresources=table(Resources,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

length(find(ismember(T_final_imp.label,'H')==1))
length(find(ismember(T_final_imp.label,'C')==1))
length(find(ismember(T_final_imp.label,'P')==1))
length(find(ismember(T_final_imp.label,'HC')==1))
length(find(ismember(T_final_imp.label,'HP')==1))
length(find(ismember(T_final_imp.label,'CP')==1))
length(find(ismember(T_final_imp.label,'HPC')==1))
dataresource_test_ind=[randsample(find(ismember(T_final_imp.label,'H')==1),2000);randsample(find(ismember(T_final_imp.label,'C')==1),2000);randsample(find(ismember(T_final_imp.label,'P')==1),2000);randsample(find(ismember(T_final_imp.label,'HC')==1),1000);find(ismember(T_final_imp.label,'HP')==1);find(ismember(T_final_imp.label,'CP')==1);find(ismember(T_final_imp.label,'HPC')==1)];
dataresource_test=T_final_imp(dataresource_test_ind,:);
dataresource_test_labels=table2array(dataresource_test(:,77));
dataresource_test_vec=T_full(dataresource_test_ind,:);
save dataresource_test_var.mat dataresource_test_ind dataresource_test dataresource_test_labels dataresource_test_vec

T_final_imp_resource_full=T_final_imp(setdiff(1:size(T_final_imp,1),dataresource_test_ind),:);
neut_delet = table2array(T_final_imp(:,77));
neut_delet_resource_full=neut_delet(setdiff(1:size(T_final_imp,1),dataresource_test_ind),1);
T_final_imp_resource_full_vec=T_final_imp_resource_full(:,[5 6 7 8 10 11 12 13 44:73 14:43]);

ind_uniprot=randsample(find(ismember(T_final_imp_resource_full.label,'H')==1),20876);
T_final_imp_resource_uniprot=T_final_imp_resource_full(ind_uniprot,[5 6 7 8 10 11 12 13 44:73 14:43]);
neut_delet_uniprot=neut_delet_resource_full(ind_uniprot,1);

ind_clinvar=[randsample(find(ismember(T_final_imp_resource_full.label,'C')==1),18620);randsample(find(ismember(T_final_imp_resource_full.label,'HC')==1),2256)];
T_final_imp_resource_clinvar=T_final_imp_resource_full(ind_clinvar,[5 6 7 8 10 11 12 13 44:73 14:43]);
neut_delet_clinvar=neut_delet_resource_full(ind_clinvar,1);

ind_pmd=randsample(find(ismember(T_final_imp_resource_full.label,'P')==1),20876);
T_final_imp_resource_pmd=T_final_imp_resource_full(ind_pmd,[5 6 7 8 10 11 12 13 44:73 14:43]);
neut_delet_pmd=neut_delet_resource_full(ind_pmd,1);

Mdl_uniprot_train=fitcensemble(T_final_imp_resource_uniprot,neut_delet_uniprot,'Method','Bag');
[testPredictions,testScores]=predict(Mdl_uniprot_train,dataresource_test_vec);
confmat=confusionmat(dataresource_test_labels,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(dataresource_test_labels==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(dataresource_test_labels,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_dataresources(1,2:11)=Perf_table_temp;

Mdl_clinvar_train=fitcensemble(T_final_imp_resource_clinvar,neut_delet_clinvar,'Method','Bag');
[testPredictions,testScores]=predict(Mdl_clinvar_train,dataresource_test_vec);
confmat=confusionmat(dataresource_test_labels,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(dataresource_test_labels==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(dataresource_test_labels,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_dataresources(2,2:11)=Perf_table_temp;

Mdl_pmd_train=fitcensemble(T_final_imp_resource_pmd,neut_delet_pmd,'Method','Bag');
[testPredictions,testScores]=predict(Mdl_pmd_train,dataresource_test_vec);
confmat=confusionmat(dataresource_test_labels,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(dataresource_test_labels==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(dataresource_test_labels,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_dataresources(3,2:11)=Perf_table_temp;

Mdl_resource_full_train=fitcensemble(T_final_imp_resource_full_vec,neut_delet_resource_full,'Method','Bag');
[testPredictions,testScores]=predict(Mdl_resource_full_train,dataresource_test_vec);
confmat=confusionmat(dataresource_test_labels,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(dataresource_test_labels==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(dataresource_test_labels,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
Perf_table_dataresources(4,2:11)=Perf_table_temp;



% Training the finalized model (Hyper-parameter optimization by cross-validation, and hold-out test):

opts = detectImportOptions('Paper_files/originaldata/training_uptodate_full.txt');
opts = setvartype(opts,opts.VariableNames(1,[13:14 17:47]),'char');
opts = setvartype(opts,opts.VariableNames(1,[9:12 15:16 48:77]),'double');
T_final = readtable('Paper_files/originaldata/training_uptodate_full.txt',opts);
neut_delet = table2array(T_final(:,78));
T_full = T_final(:,[9:12 14:77]);

tic
n = size(T_full,1);
m = floor(log(n - 1)/log(3));
maxNumSplits = 3.^(0:m);
maxNumSplits = [maxNumSplits([2 5 8]) size(T_full,1)-1];
numMNS = numel(maxNumSplits);
numTrees = [50 150 300 500];
numT = numel(numTrees);
numvartosamp = [2 8 24 size(T_full,2)];
numV = numel(numvartosamp);
Mdl_rf_hypopt = cell(numT,numMNS,numV);
AUC=zeros(64,1);Recall=zeros(64,1);Precision=zeros(64,1);F1score=zeros(64,1);Accuracy=zeros(64,1);MCC=zeros(64,1);TP=zeros(64,1);FP=zeros(64,1);FN=zeros(64,1);TN=zeros(64,1);
NumVartoSam=repmat(numvartosamp',16,1);
maxNumSplit=repmat([repmat(maxNumSplits(1,1),4,1);repmat(maxNumSplits(1,2),4,1);repmat(maxNumSplits(1,3),4,1);repmat(maxNumSplits(1,4),4,1)],4,1);
numTree=[repmat(numTrees(1,1),16,1);repmat(numTrees(1,2),16,1);repmat(numTrees(1,3),16,1);repmat(numTrees(1,4),16,1)];
Perf_table_hyperpar=table(numTree,maxNumSplit,NumVartoSam,AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
to=0;
for i = 1:numT
    for j = 1:numMNS
        for k = 1:numV
            to=to+1;
            t = templateTree('MaxNumSplits',maxNumSplits(j),'NumVariablesToSample',numvartosamp(k));
            disp(['numTrees: ', num2str(numTrees(i)), ', maxNumSplit: ', num2str(maxNumSplits(j)), ', NumVariablesToSample: ', num2str(numvartosamp(k))])
            Mdl_rf_hypopt = fitcensemble(T_full,neut_delet,'Method','Bag','CrossVal','on','NumLearningCycles',numTrees(i),'Learners',t,'KFold',5);
            
            [validationPredictions,validationScores]=kfoldPredict(Mdl_rf_hypopt);
            confmat=confusionmat(Mdl_rf_hypopt.Y,validationPredictions);
            confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
            [~,~,~,AUC]=perfcurve(Mdl_rf_hypopt.Y,validationScores(:,2),1);
            TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
            Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
            Perf_table_temp=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);
            Perf_table_hyperpar(to,4:13)=Perf_table_temp;
        end
    end
end
toc

test_ind=unique(randperm(length(neut_delet),10000))';
T_full_train=T_full(setdiff(1:length(neut_delet),test_ind)',:);
neut_delet_train=neut_delet(setdiff(1:length(neut_delet),test_ind)',:);
T_full_test=T_full(test_ind,:);
neut_delet_test=neut_delet(test_ind,:);
tic
t = templateTree('MaxNumSplits',96273,'NumVariablesToSample',8);
Mdl_full_test=fitcensemble(T_full_train,neut_delet_train,'Method','Bag','NumLearningCycles',300,'Learners',t);
toc
tic
[testPredictions,testScores]=predict(Mdl_full_test,T_full_test);
toc
confmat=confusionmat(neut_delet_test,testPredictions);
confmat_fix=[confmat(2,2) confmat(2,1);confmat(1,2) confmat(1,1)];
confmat_fix(1,1)
length(find(neut_delet_test==1 & testPredictions==1))
[~,~,~,AUC]=perfcurve(neut_delet_test,testScores(:,2),1);
TP=confmat_fix(1,1);TN=confmat_fix(2,2);FN=confmat_fix(1,2);FP=confmat_fix(2,1);
Recall=TP/(TP+FN);Precision=TP/(TP+FP);F1score=(2*TP)/(2*TP+FP+FN);Accuracy=(TP+TN)/(TP+FP+FN+TN);MCC=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^(1/2));
Perf_table_test_finalized=table(AUC,Recall,Precision,F1score,Accuracy,MCC,TP,FN,FP,TN);

t = templateTree('MaxNumSplits',96273,'NumVariablesToSample',8);
Mdl_full_full=fitcensemble(T_full,neut_delet,'Method','Bag','NumLearningCycles',300,'Learners',t);
imp = predictorImportance(Mdl_full_full);
imp_scaled=(1/max(imp))*imp;
[imp_scaled_sort,idx]=sort(imp_scaled,'ascend');
figure
barh(imp_scaled_sort)
grid on
title('All Features - Predictor Importance Estimates')
xlabel('Scaled importance')
ylabel('Predictor variable')
set(gca, 'YTick', linspace(1,100), 'YTickLabels', Mdl_full_test.PredictorNames(idx), 'TickLabelInterpreter', 'none')
axis([0 1 0 68.95])



%Saving variables (models sometimes too large to be saved in the default format):
save Perf_table_ablation.mat Perf_table_ablation
save Perf_table_classifiers.mat Perf_table_classifiers
save Perf_table_dataresources.mat Perf_table_dataresources
save Perf_table_hyperpar.mat Perf_table_hyperpar
save Perf_table_test_finalized.mat Perf_table_test_finalized

save Mdl_rf_hypopt.mat Mdl_rf_hypopt
save Mdl_annobin.mat Mdl_annobin
save Mdl_annobindis.mat Mdl_annobindis
save Mdl_domain.mat Mdl_domain
save Mdl_full.mat Mdl_full
save Mdl_location.mat Mdl_location
save Mdl_physico.mat Mdl_physico
save Mdl_strucannobin.mat Mdl_strucannobin
save Mdl_structural.mat Mdl_structural
save Mdl_full_full_feature_importance_model.mat Mdl_full_full

save Mdl_full_500.mat Mdl_full_500
save Mdl_nbayes_full.mat Mdl_nbayes_full
save Mdl_svm_full.mat Mdl_svm_full
save Mdl_adaboost_full.mat Mdl_adaboost_full
save Mdl_logitboost_full.mat Mdl_logitboost_full

save Mdl_uniprot_train.mat Mdl_uniprot_train
save Mdl_clinvar_train.mat Mdl_clinvar_train
save Mdl_pmd_train.mat Mdl_pmd_train
save Mdl_resource_full_train.mat Mdl_resource_full_train

save T_final_imp.mat T_final_imp
save T_full.mat T_full
