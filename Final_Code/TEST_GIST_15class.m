% Athanasios Tsiligkaridis, April 2nd, 2017
% Testing ANN's on 15 Scene dataset using gist features
clear, clc, close all
%% Load 2 Class (Indoor/Outdoor) Data
fprintf('Loading Indoor/Outdoor Data with GIST features. \n')
load('X_gist.mat');
inputs = X_gist'; %512 features
load('Y.mat')
targets = zeros(15,4485);
for i = 1:4485
   targets(Y(i),i)=1; 
end

net = feedforwardnet([200 200],'trainscg');
net = init(net);

% judge performance using mse
net.performFcn = 'mse';

% data partitioning
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

net = configure(net,inputs,targets);

% training
[net,tr] = train(net,inputs,targets);

% testing
outputs = net(inputs);

[~,empirical_labels]=max(outputs);
disp(confusionmat(Y',empirical_labels))
CCR_val = sum(Y'==empirical_labels)/length(Y)

%% test new data:
% 
% %%%%%%%%%%%%%%%%%%%%%%%%% outdoor 1
fprintf('Bedroom Image (1): \n')
% Load image
img = imread('fifteen_bedroom_1.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
bedroom = net(gist')

%%%%%%%%%%%%%%%%% outdoor - 2
fprintf('Forest Image (7): \n')
% Load image
img = imread('fifteen_forest_1.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
forest = net(gist')

%%%%%%%%%%%%%%%%%%% indoor - 1
fprintf('Skyline (13): \n')
% Load image
img = imread('fifteen_skyline_1.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
skyline = net(gist')

fprintf('Beach (6): \n')
% Load image
img = imread('fifteen_beach_1.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
beach = net(gist')

fprintf('Living Room (5): \n')
% Load image
img = imread('fifteen_livingroom_1.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
livingroom = net(gist')

