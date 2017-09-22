% Athanasios Tsiligkaridis, April 2nd, 2017
% Testing ANN's on 15 Scene dataset using centrist features
clear, clc, close all
%% Load 2 Class (Indoor/Outdoor) Data
fprintf('Loading Indoor/Outdoor Data with CENBTRISR features. \n')
load('X_cent.mat');
inputs = X_cent'; %512 features
load('Y.mat')
targets = zeros(15,4485);
for i = 1:4485
   targets(Y(i),i)=1; 
end

net = feedforwardnet([200 200],'trainscg');
net = init(net);
net.layers{3}.transferFcn = 'tansig';

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
disp(confusionmat((Y'),empirical_labels))
CCR_val = sum((Y')==empirical_labels)/length(Y')

%% test new data:
img = imread('fifteen_bedroom_1.jpg'); X_test = centrist(img); bedroom = net(X_test')

img = imread('fifteen_beach_1.jpg'); X_test = centrist(img); beach = net(X_test')

img = imread('fifteen_skyline_1.jpg'); X_test = centrist(img); skyline = net(X_test')

img = imread('fifteen_forest_1.jpg'); X_test = centrist(img); forest = net(X_test')






