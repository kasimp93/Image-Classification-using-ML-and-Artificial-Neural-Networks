% Kasim Patel
% 4/2/2017
% 15-Scene Data Set SVM-classification
% EC-503 Project

clear all;
close all;
clc;

rng('default');

%% load data 15scene
load('Data15\\X_cent.mat')
load('Data15\\X_gist.mat');
load('Data15\\Y.mat');
Xg = X_gist;
Xc = X_cent;

%% split data 70-30 for training and testing
[tr_idx, te_idx] = crossvalind('HoldOut', length(Y), 0.3);

trXg = Xg(tr_idx,:); %tr gist vectors
trXc = Xc(tr_idx,:); %tr centrist vectors
trY = Y(tr_idx);     %tr labels

teXg = Xg(te_idx,:); %te gist vectors
teXc = Xc(te_idx,:); %te centrist vectors
teY = Y(te_idx);     %te labels

%% SVM-training
disp('training');
meanXg = mean(trXg);
stdXg = std(trXg);
meanXc = mean(trXc);
stdXc = std(trXc);
for i = 1:size(trXg,1)
   trXg(i,:) = (trXg(i,:) - meanXg)./stdXg;
   trXc(i,:) = (trXc(i,:) - meanXc)./stdXc;
end


%% Old One
mdl_comb = fitcecoc([trXg,trXc],trY,'Coding','onevsone','Learners','svm');
mdl_cent = fitcecoc(trXc,trY,'Coding','onevsone','Learners','svm');
mdl_gist = fitcecoc(trXg,trY,'Coding','onevsone','Learners','svm');

% %% Optimizing Hyperparameters
% mdl_gist = fitcecoc(trXg,trY,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
% mdl_cent = fitcecoc(trXc,trY,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
% z = [trXg,trXc];
% mdl_comb = fitcecoc(z,trY,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

% %% Old One
% mdl_comb = fitcecoc([trXg,trXc],trY,'Coding','onevsall','Learners','svm','CrossVal','on');
% mdl_cent = fitcecoc(trXc,trY,'Coding','onevsall','Learners','svm','CrossVal','on');
% mdl_gist = fitcecoc(trXg,trY,'Coding','onevsall','Learners','svm','CrossVal','on');

%% fitcsvm with all c and sigma
% 
% power = -5:15;
% c = 2.^power;
% 
% pow_s = -13:3
% sigma = 2.^pow_s ;
% 
% for i = 1:length(c)
%     for j = 1:length(sigma)
%         n = size(trY,1);
%         yu= ones(n,1);
%         C = c(k) * yu;
%         mdl_comb(i) = fitcsvm([trXg,trXc],trY,'BoxConstraint',C,'KernelFunction','linear','autoscale',false,'CrossVal','on');
%         mdl_cent(i) = fitcsvm(trXc,trY,'BoxConstraint',C,'KernelFunction','linear','autoscale', false,'CrossVal','on');
%         mdl_gist(i) = fitcsvm(trXg,trY,'BoxConstraint',C,'KernelFunction','linear','autoscale', false,'CrossVal','on');
% end
% end

%% fitcecoc with all c and sigma
% 
% power = -5:15;
% c = 2.^power;
% 
% pow_s = -13:3
% sigma = 2.^pow_s ;
% 
% for i = 1:length(c)
%     for j = 1:length(sigma)
%         n = size(trY,1);
%         yu= ones(n,1);
%         C = c(k) * yu;
%         mdl_comb(i) = fitcecoc([trXg,trXc],trY,'Coding','onevsall','Learners','svm',{'BoxConstraint",'KernelScale'},'OptimizeHyperparameters','auto','CrossVal','on');
%         mdl_cent(i) = fitcecoc(trXc,trY,'Coding','onevsall','Learners','svm',{'BoxConstraint",'KernelScale'},'OptimizeHyperparameters','auto','CrossVal','on');
%         mdl_gist(i) = fitcecoc(trXg,trY,'Coding','onevsall','Learners','svm',{'BoxConstraint",'KernelScale'},'OptimizeHyperparameters','auto','CrossVal','on');
% end
% end

%% SVM-prediction
disp('classifying');
for i = 1:size(teXg,1)
   teXg(i,:) = (teXg(i,:) - meanXg)./stdXg;
   teXc(i,:) = (teXc(i,:) - meanXc)./stdXc;
end


Yhat_comb_te = predict(mdl_comb,[teXg,teXc]);
Yhat_gist_te = predict(mdl_gist,teXg);
Yhat_cent_te = predict(mdl_cent,teXc);

Yhat_comb_tr = predict(mdl_comb,[trXg,trXc]);
Yhat_gist_tr = predict(mdl_gist,trXg);
Yhat_cent_tr = predict(mdl_cent,trXc);

%% performance analysis
confmat_gist = confusionmat(teY,Yhat_gist_te,'order',1:15);
confmat_cent = confusionmat(teY,Yhat_cent_te,'order',1:15);
confmat_comb = confusionmat(teY,Yhat_comb_te,'order',1:15);

ccr_gist_te = sum(Yhat_gist_te == teY)/length(teY)
ccr_cent_te = sum(Yhat_cent_te == teY)/length(teY)
ccr_comb_te = sum(Yhat_comb_te == teY)/length(teY)

ccr_gist_tr = sum(Yhat_gist_tr == trY)/length(trY)
ccr_cent_tr = sum(Yhat_cent_tr == trY)/length(trY)
ccr_comb_tr = sum(Yhat_comb_tr == trY)/length(trY)

cp_gist = classperf(teY, Yhat_gist_te);
cp_cent = classperf(teY, Yhat_cent_te);
cp_comb = classperf(teY, Yhat_comb_te);