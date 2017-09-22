% knn-classification

%% load data 15scene
load('Data15\\X_cent.mat')
load('Data15\\X_gist.mat');
load('Data15\\X_comb.mat');
load('Data15\\Y.mat');
Xg = X_gist;
Xc = X_cent;
Xcom = X_comb;

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

%% knn-training
disp('training');
i = 1;
for k = 1:5:100
    mdl_gist = fitcknn(trXg,trY,'NumNeighbors',k,'BreakTies','random','Distance','euclidean');
    mdl_cent = fitcknn(trXc,trY,'NumNeighbors',k,'BreakTies','random','Distance','euclidean');
    mdl_comb = fitcknn(trXcom,trY,'NumNeighbors',k,'BreakTies','random','Distance','euclidean');
    
    %% knn-prediction
    fprintf('classifying for k=%d\n',k);
    Yhat_gist = predict(mdl_gist,teXg);
    Yhat_cent = predict(mdl_cent,teXc);
    Yhat_comb = predict(mdl_comb,teXcom);
    
    %% performance analysis
    confmat_gist = confusionmat(teY,Yhat_gist,'order',1:15);
    confmat_cent = confusionmat(teY,Yhat_cent,'order',1:15);
    confmat_comb = confusionmat(teY,Yhat_comb,'order',1:15);
    ccr_gist(i) = (sum(Yhat_gist == teY)/length(teY))*100;
    ccr_cent(i) = (sum(Yhat_cent == teY)/length(teY))*100;
    ccr_comb(i) = (sum(Yhat_comb == teY)/length(teY))*100;
    i = i+1;
end

k_vec = 1:5:100;
%% plot results
plot(k_vec,ccr_gist,'-*');
ylim([0 100])
grid on;
hold on;
plot(k_vec,ccr_cent,'-*');
plot(k_vec,ccr_comb,'-*');
hold off;
ylabel('CCR (%)');
xlabel('k');
title('15-scene Classification with kNN (euclidean)- CCR vs k');
legend('GIST','CENTRIST','GIST+CENTRIST');

%% get best ccr for each gist,cent,&comb and corresponding lambda*
[gist_bestccr,gist_idx] = max(ccr_gist);
gist_bestk = k_vec(gist_idx);

[cent_bestccr,cent_idx] = max(ccr_cent);
cent_bestk = k_vec(cent_idx);

[comb_bestccr,comb_idx] = max(ccr_comb);
comb_bestk = k_vec(comb_idx);

gist_bestccr
gist_bestk
cent_bestccr
cent_bestk
comb_bestccr
comb_bestk