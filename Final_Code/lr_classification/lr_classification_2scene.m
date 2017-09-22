% lr-classification

%% load data 15scene
load('Data2\\X_cent2.mat')
load('Data2\\X_gist2.mat');
load('Data2\\X_comb2.mat');
load('Data2\\Y2.mat');
Xg = X_gist2;
Xc = X_cent2;
Xcom = X_comb2;
Y = Y2+1; %logistic regression needs to have +ve category numbers
          %so instead of 0&1, we make it 1&2

%% split data 70-30 for training and testing
[tr_idx, te_idx] = crossvalind('HoldOut', length(Y), 0.3);

trXg = Xg(tr_idx,:); %tr gist vectors
trXc = Xc(tr_idx,:); %tr centrist vectors
trXcom = Xcom(tr_idx,:); %tr combined vectors
trY = Y(tr_idx);     %tr labels

teXg = Xg(te_idx,:); %te gist vectors
teXc = Xc(te_idx,:); %te centrist vectors
teXcom = Xcom(te_idx,:); %te combined vectors
teY = Y(te_idx);     %te labels

%% lr-training
disp('training');
k = 50;
tic
b_gist = mnrfit(trXg,trY);
gist_time = toc
tic
b_cent = mnrfit(trXc,trY);
cent_time = toc
tic
b_comb = mnrfit(trXcom,trY);
comb_time = toc

%% lr-prediction
disp('classifying');
tic
pihat_gist = mnrval(b_gist,teXg);
pihat_cent = mnrval(b_cent,teXc);
pihat_comb = mnrval(b_comb,teXcom);
predict_time = toc

[~,Yhat_gist] = max(pihat_gist,[],2);
[~,Yhat_cent] = max(pihat_cent,[],2);
[~,Yhat_comb] = max(pihat_comb,[],2);
%% performance analysis
confmat_gist = confusionmat(teY,Yhat_gist,'order',1:2);
confmat_cent = confusionmat(teY,Yhat_cent,'order',1:2);
confmat_comb = confusionmat(teY,Yhat_comb,'order',1:2);
ccr_gist = sum(Yhat_gist == teY)/length(teY)
ccr_cent = sum(Yhat_cent == teY)/length(teY)
ccr_comb = sum(Yhat_comb == teY)/length(teY)

cp_gist = classperf(teY, Yhat_gist);
cp_cent = classperf(teY, Yhat_cent);
cp_comb = classperf(teY, Yhat_comb);

%% plot results
bar([ccr_gist,ccr_cent,ccr_comb]*100);
ylim([0 100])
set(gca,'XTickLabel',{'GIST', 'CENTRIST', 'GIST+CENTRIST'})
ylabel('CCR (%)');
xlabel('Feature Used');
title('2-scene Classification with Logistic Regression - CCR vs Feature Used');
