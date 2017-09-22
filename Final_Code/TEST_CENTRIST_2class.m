% Athanasios Tsiligkaridis, April 2nd, 2017
% Testing ANN's on 2 class dataset with centrist features
clear, clc, close all
%% Load 2 Class (Indoor/Outdoor) Data
fprintf('Loading Indoor/Outdoor Data with CENTRIST features. \n')
load('X_cent2.mat');
inputs = X_cent2'; %512 features
load('Y2.mat')
targets = zeros(2,length(Y2));
for i = 1:length(Y2)
   targets(Y2(i)+1,i)=1; 
end

net = feedforwardnet([200 200],'trainscg');
net = init(net);
net.layers{3}.transferFcn = 'softmax';

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
disp(confusionmat((Y2'+1),empirical_labels))
CCR_val = sum((Y2'+1)==empirical_labels)/length(Y2')

%% test new data:
img = imread('outdoor_ex.jpg'); X_test = centrist(img); outdoor_ex = net(X_test')

img = imread('outdoor_car.jpg'); X_test = centrist(img); outdoor_car = net(X_test')

img = imread('outdoor_NOLA.jpg'); X_test = centrist(img); outdoor_nola = net(X_test')

img = imread('outdoor_sunnyday.jpg'); X_test = centrist(img); outdoor_sunnyday = net(X_test')

img = imread('outdoor_airplane.jpg'); X_test = centrist(img); outdoor_airplane = net(X_test')

img = imread('indoor_bathroom.jpg'); X_test = centrist(img); indoor_bathroom = net(X_test')

img = imread('indoor_ex.jpg'); X_test = centrist(img); indoor_livingroom = net(X_test')

img = imread('indoor_starbucks.jpg'); X_test = centrist(img); indoor_starbucks = net(X_test')

img = imread('indoor_store.jpg'); X_test = centrist(img); indoor_store = net(X_test')







