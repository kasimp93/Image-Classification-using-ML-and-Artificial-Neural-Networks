% Kasim Patel
% 4/2/2017
% 2-Scene Data Set SVM-classification
% EC-503 Project

clear all;
close all;
clc;

rng('default');

%% load data 15scene
load('Data21\\X_cent2.mat')
load('Data21\\X_gist2.mat');
load('Data21\\Y2.mat');
Xg = X_gist2;
Xc = X_cent2;

%% split data 70-30 for training and testing
[tr_idx, te_idx] = crossvalind('HoldOut', length(Y2), 0.3);

trXg = Xg(tr_idx,:); %tr gist vectors
trXc = Xc(tr_idx,:); %tr centrist vectors
trY = Y2(tr_idx);     %tr labels

teXg = Xg(te_idx,:); %te gist vectors
teXc = Xc(te_idx,:); %te centrist vectors
teY = Y2(te_idx);     %te labels

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



K = 5;
Indices = crossvalind('Kfold',trY,K);

pow_b = -5:15
b_c = 2.^pow_b ;

pow_s = -13:3
sigma = 2.^pow_s ;

for i = 1:length(b_c)
    for j = 1:length(sigma)
        for k = 1:K
            test = (Indices == k);
            train = ~test;
            %z = [trXg,trXc];
            X_train = trXg(train,:);
            X_test = trXg(test,:);
            no_train = size(X_train,1);
            no_test = size(X_test,1);
            Y_train = trY(train);
            Y_test = trY(test);
            C = b_c(i) * ones(no_train,1)
            SVMmodel = svmtrain(X_train,Y_train, 'kernel_function','rbf','rbf_sigma', sigma(j), 'boxconstraint',C, 'autoscale','false');
            Y_predict = svmclassify(SVMmodel,X_test);
            CCR(k) = sum(Y_predict == Y_test)/no_test;
        end
            avgCCR(i,j) = mean(CCR);
    end
end

figure; 
contourf( log(sigma), log(b_c), avgCCR); 
colorbar;
title('Graph of log(Sigma), log(Box Constraint) with avgCCR (GIST)')
xlabel('Sigma') % x-axis label
ylabel('Box Consraint')



CCR = max(avgCCR(:));

