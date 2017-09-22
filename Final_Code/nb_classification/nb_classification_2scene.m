% nb-classification

%% load data 15scene
load('Data2\\X_cent2.mat')
load('Data2\\X_gist2.mat');
load('Data2\\X_comb2.mat');
load('Data2\\Y2.mat');
Xg = X_gist2;
Xc = X_cent2;
Xcom = X_comb2;
Y = Y2;

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

%% nb-training
disp('training');
k = 50;
mdl_gist = fitcnb(trXg,trY);
mdl_cent = fitcnb(trXc,trY);
mdl_comb = fitcnb(trXcom,trY);

%% nb-prediction
disp('classifying');
Yhat_gist = predict(mdl_gist,teXg);
Yhat_cent = predict(mdl_cent,teXc);
Yhat_comb = predict(mdl_comb,teXcom);

%% performance analysis
confmat_gist = confusionmat(teY,Yhat_gist,'order',0:1);
confmat_cent = confusionmat(teY,Yhat_cent,'order',0:1);
confmat_comb = confusionmat(teY,Yhat_comb,'order',0:1);
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
title('15-scene Classification with Nive Bayes - CCR vs Feature Used');
