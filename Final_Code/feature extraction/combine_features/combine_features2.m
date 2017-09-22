%% load data 15scene
load('dataset\\X_cent2.mat')
load('dataset\\X_gist2.mat');

%% standardize features (subtract mean and div by variance)
disp('standardizing');
meanXggist = mean(X_gist2);
stdXgist = std(X_gist2);
meanXcent = mean(X_cent2);
stdXcent = std(X_cent2);
for i = 1:22
   X_gist(i,:) = (X_gist(i,:) - meanXggist)./stdXgist;
   X_cent(i,:) = (X_cent(i,:) - meanXcent)./stdXcent;
end

%% combine features after standardization
X_comb = [X_gist,X_cent];
save('X_comb','X_comb');
