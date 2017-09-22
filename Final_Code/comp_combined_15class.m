% For a test image, get combined feature vector.
function featurevec = comp_combined_15class(img)
%% load means of 2 class data 
% load('X_cent.mat')
% load('X_gist.mat');
% 
% %% standardize features (subtract mean and div by variance)
% disp('standardizing test examples');
% meanXggist = mean(X_gist)
% stdXgist = std(X_gist)
% meanXcent = mean(X_cent)
% stdXcent = std(X_cent)
% for i = 1:size(X_gist2,1)
%    X_gist2(i,:) = (X_gist2(i,:) - meanXggist)./stdXgist;
%    X_cent2(i,:) = (X_cent2(i,:) - meanXcent)./stdXcent;
% end
load('stats_15class.mat')
%% Get image stuff: - from open source code
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
disp('gist...')
[gistfeatures, param] = LMgist(img, '', param);
disp('centrist...')
centristfeatures = centrist(img);

gistfeatures     = (gistfeatures - meanXggist)./stdXgist;
centristfeatures = (centristfeatures - meanXcent)./stdXcent;

featurevec = [gistfeatures,centristfeatures];
end