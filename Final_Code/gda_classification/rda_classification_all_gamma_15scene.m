% gda-classification

%% load data 15scene
load('Data15\\X_cent.mat')
load('Data15\\X_gist.mat');
load('Data15\\X_comb.mat');
load('Data15\\Y.mat');
numofClass = 15;
Xg = X_gist;
Xc = X_cent;
Xcom= X_comb;

%% split data 70-30 for training and testing
[tr_idx, te_idx] = crossvalind('HoldOut', length(Y), 0.3);

trXg = Xg(tr_idx,:);     %tr gist vectors
trXc = Xc(tr_idx,:);     %tr centrist vectors
trXcom = Xcom(tr_idx,:); %tr combined vectors
trY = Y(tr_idx);         %tr labels

teXg = Xg(te_idx,:);     %te gist vectors
teXc = Xc(te_idx,:);     %te centrist vectors
teXcom = Xcom(te_idx,:); %te combined vectors
teY = Y(te_idx);         %te labels

%% qda-training
disp('training');

i = 1;
for gamma = 0:0.1:1
    mdl_gist = fitcdiscr(trXg,trY,'DiscrimType','linear','Gamma',gamma);
    mdl_cent = fitcdiscr(trXc,trY,'DiscrimType','linear','Gamma',gamma);
    mdl_comb = fitcdiscr(trXcom,trY,'DiscrimType','linear','Gamma',gamma);
    
    
    %% qda-prediction
    disp('classifying');
    Yhat_gist = predict(mdl_gist,teXg);
    Yhat_cent = predict(mdl_cent,teXc);
    Yhat_comb = predict(mdl_comb,teXcom);
    
    %% performance analysis
    confmat_gist = confusionmat(teY,Yhat_gist,'order',1:15);
    confmat_cent = confusionmat(teY,Yhat_cent,'order',1:15);
    confmat_comb = confusionmat(teY,Yhat_comb,'order',1:15);
    ccr_gist(i) = 100*(sum(Yhat_gist == teY)/length(teY));
    ccr_cent(i) = 100*(sum(Yhat_cent == teY)/length(teY));
    ccr_comb(i) = 100*(sum(Yhat_comb == teY)/length(teY));
    i = i+1;
end

lambda = 0:0.1:1;

%% plot results
plot(lambda,ccr_gist,'-*');
ylim([0 100])
grid on;
hold on;
plot(lambda,ccr_cent,'-*');
plot(lambda,ccr_comb,'-*');
hold off;
ylabel('CCR (%)');
xlabel('\lambda (regularization param)');
title('15-scene Classification with Regularized LDA - CCR vs \lambda');
legend('GIST','CENTRIST','GIST+CENTRIST');

%% get best ccr for each gist,cent,&comb and corresponding lambda*
[gist_bestccr,gist_idx] = max(ccr_gist);
gist_bestlambda = lambda(gist_idx);

[cent_bestccr,cent_idx] = max(ccr_cent);
cent_bestlambda = lambda(cent_idx);

[comb_bestccr,comb_idx] = max(ccr_comb);
comb_bestlambda = lambda(comb_idx);