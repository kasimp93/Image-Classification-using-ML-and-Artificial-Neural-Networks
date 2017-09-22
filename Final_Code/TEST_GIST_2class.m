% Athanasios Tsiligkaridis, April 2nd, 2017
% Testing ANN's on 2 Scene dataset using gist features
clear, clc, close all
%% Load 2 Class (Indoor/Outdoor) Data
fprintf('Loading Indoor/Outdoor Data with GIST features. \n')
load('X_gist2.mat');
inputs = X_gist2'; %512 features
targets = zeros(2,2386);
targets(1,1:1193) = 1;   % indoor -> [1;0]
targets(2,1194:end) = 1; % outdoor -> [0;1]
targets_run = zeros(1,2386);
targets_run(1,1:1193)=1;
targets_run(1,1194:end)=2;

% 2 layers, 200 neurons
net = feedforwardnet([200 200],'trainscg');
net = init(net);
net.layers{1}.transferFcn = 'logsig'; 
net.layers{2}.transferFcn = 'logsig'; 
net.layers{3}.transferFcn = 'logsig';

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
disp(confusionmat(targets_run,empirical_labels))
CCR_val = sum(targets_run==empirical_labels)/length(targets_run)

%% test new data:

%%%%%%%%%%%%%%%%%%%%%%%%% outdoor 1
fprintf('Outdoor image 1: \n')
% Load image
img = imread('outdoor_ex.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
output_NOLA = net(gist')

%%%%%%%%%%%%%%%%% outdoor - 2
fprintf('Outdoor image 2: \n')
% Load image
img = imread('outdoor_hotel.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
output_hotel = net(gist')

%%%%%%%%%%%%%%%%%%% indoor - 1
fprintf('Indoor image 1: \n')
% Load image
img = imread('indoor_ex.jpg');

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
[gist, param] = LMgist(img, '', param);
output_livingroom = net(gist')

