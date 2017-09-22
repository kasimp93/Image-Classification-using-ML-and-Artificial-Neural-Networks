%TEST_COMBINED_15class
% Athanasios Tsiligkaridis, April 2nd, 2017
% Testing ANN's on 15 Scene dataset using combined features
clear, clc, close all
%% Load 2 Class (Indoor/Outdoor) Data
fprintf('Loading Indoor/Outdoor Data with CENTRIST features. \n')
load('X_comb.mat');
inputs = X_comb'; %512 features
load('Y.mat')
targets = zeros(2,length(Y));
for i = 1:length(Y)
   targets(Y(i),i)=1; 
end

net = feedforwardnet([200 200],'trainscg');
net = init(net);

% performance - mse
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
disp(confusionmat((Y'),empirical_labels))
CCR_val = sum((Y')==empirical_labels)/length(Y')

%% test new data:
img = imread('fifteen_bedroom_1.jpg'); bedroom = abs(net(comp_combined_15class(img)'))

img = imread('fifteen_beach_1.jpg'); beach = abs(net(comp_combined_15class(img)'))

img = imread('fifteen_skyline_1.jpg'); skyline = abs(net(comp_combined_15class(img)'))

img = imread('fifteen_forest_1.jpg'); forest = abs(net(comp_combined_15class(img)'))






