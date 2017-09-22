% For a test image, get combined feature vector.
function featurevec = comp_combined_2class(img)
%% load means of 2 class data 
% load('X_cent2.mat');
% load('X_gist2.mat');
% 
% %% standardize features (subtract mean and div by variance)
% disp('standardizing test examples');
% meanXggist = mean(X_gist2);
% stdXgist = std(X_gist2);
% meanXcent = mean(X_cent2);
% stdXcent = std(X_cent2);
% for i = 1:size(X_gist2,1)
%    X_gist2(i,:) = (X_gist2(i,:) - meanXggist)./stdXgist;
%    X_cent2(i,:) = (X_cent2(i,:) - meanXcent)./stdXcent;
% end
load('stats_2class.mat')

%% Get image stuff: - from OPEN SOURCE CODE
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