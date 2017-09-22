%% load data 15scene
load('dataset\\X_cent2.mat')
load('dataset\\X_gist2.mat');

%% standardize features (subtract mean and div by variance)
disp('standardizing');
meanXggist = mean(X_gist2);
stdXgist = std(X_gist2);
meanXcent = mean(X_cent2);
stdXcent = std(X_cent2);
for i = 1:size(X_gist2,1)
   X_gist2(i,:) = (X_gist2(i,:) - meanXggist)./stdXgist;
   X_cent2(i,:) = (X_cent2(i,:) - meanXcent)./stdXcent;
end

%% combine features after standardization
X_comb2 = [X_gist2,X_cent2];
save('X_comb2','X_comb2');
