%TEST_COMBINED_2class
% Athanasios Tsiligkaridis, April 2nd, 2017
% Testing ANN's on 2 class dataset - combined features
clear, clc, close all
%% Load 2 Class (Indoor/Outdoor) Data
fprintf('Loading Indoor/Outdoor Data with CENTRIST features. \n')
load('X_comb2.mat');
inputs = X_comb2'; %512 features
load('Y2.mat')
targets = zeros(2,length(Y2));
for i = 1:length(Y2)
   targets(Y2(i)+1,i)=1; 
end

net = feedforwardnet([200 200],'trainscg');
net = init(net);
net.layers{3}.transferFcn = 'softmax';

% performance - use mse
net.performFcn = 'mse';

% Data partitioning
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

net = configure(net,inputs,targets);

% training
[net,tr] = train(net,inputs,targets);

% testing
outputs = net(inputs);

[~,empirical_labels]=max(outputs);
disp(confusionmat((Y2'+1),empirical_labels))
CCR_val = sum((Y2'+1)==empirical_labels)/length(Y2')

%% test new data:
img = imread('outdoor_ex.jpg'); outdoor_ex = net(comp_combined_2class(img)')

img = imread('outdoor_car.jpg'); outdoor_car = net(comp_combined_2class(img)')

img = imread('outdoor_NOLA.jpg'); outdoor_nola = net(comp_combined_2class(img)')

img = imread('outdoor_sunnyday.jpg'); outdoor_sunnyday = net(comp_combined_2class(img)')

img = imread('outdoor_airplane.jpg'); outdoor_airplane = net(comp_combined_2class(img)')

img = imread('indoor_bathroom.jpg'); indoor_bathroom = net(comp_combined_2class(img)')

img = imread('indoor_ex.jpg'); indoor_ex = net(comp_combined_2class(img)')

img = imread('indoor_starbucks.jpg'); starbucks = net(comp_combined_2class(img)')

img = imread('indoor_store.jpg'); store = net(comp_combined_2class(img)')

img = imread('funny_airplane.jpg'); airplane = net(comp_combined_2class(img)')

img = imread('mr_bean_indoor.jpg'); beanindoor = net(comp_combined_2class(img)')








