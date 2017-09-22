clear all
close all
clc

% The first part you need to choose which Data set and which feature
% extraction method you want to train and test.
% % % % %% load the data here
%% Gist
% % % 15 classes Gist
% Data = load('X_gist.mat');
% Data_original=Data.X_gist;
% labels=load('labels.mat');
% labels=labels.labels;
% [~, number_labels]=max(labels,[],2);
% 


% % %  2 classes Gist
Data = load('gist2_2class.mat'); 
Data_original=Data.gist2;         %data points X features
labels=load('labels_2class.mat');
labels=labels.xTargets';           %data points x number_of_classes
[~, number_labels]=max(labels,[],2);

%% Comb
% % % 15 classes comb
% Data = load('X_comb.mat');
% Data_original=Data.X_comb;
% labels=load('labels.mat');
% labels=labels.labels;
% [~, number_labels]=max(labels,[],2);

% % % 2 classes comb
% Data = load('X_comb2.mat');
% Data_original=Data.X_comb2;
% labels=load('labels_2class.mat');
% labels=labels.xTargets';
% [~, number_labels]=max(labels,[],2);

%% Centrist
% % %  2 classes Cent
% Data = load('X_cent2.mat'); 
% Data_original=Data.X_cent2;         %data points X features
% labels=load('labels_2class.mat');
% labels=labels.xTargets';           %data points x number_of_classes
% [~, number_labels]=max(labels,[],2);

% % % 15 classes cent
% Data = load('X_cent.mat');
% Data_original=Data.X_cent;
% labels=load('labels.mat');
% labels=labels.labels;
% [~, number_labels]=max(labels,[],2);


%% Network configuration
%eps=10^(-4); I used to check if the gradient was correct and i used this
%value eps
eps2=2; % This is used for the random initialization of the weights matrix

alpha=10^(-1);        %gradient step
L=4;                  % Number of layers
OutputNodes=size(labels,2);
Hidden_L=L-2;

S=size(Data_original,2); % number of nodes for input
Sh=200; % number of nodes for hidden layers

W1=eps2*rand(Sh,S+1)-eps2/2;
W2=eps2*rand(Sh,Sh+1)-eps2/2;
W3=eps2*rand(OutputNodes,Sh+1)-eps2/2;
W=[W1;[W2;W3] zeros(OutputNodes+Sh,S-(Sh))]; %Weights matrix
S_vec=[S Sh Sh OutputNodes]; % 2 hidden layers

input=zeros(L,S+1);
output=zeros(size(Data_original,1),OutputNodes);
bias=1;
m=size(Data_original,1); % number of data points
%classes=size(labels,2);  %Number of Classes in your dataset


random_indices=randperm(m);
Data_rand_indices=Data_original(random_indices,:);
labels_rand=labels(random_indices,:);

Data_Train_valid=Data_rand_indices(1:floor(0.85*size(Data_original,1)),:);
labels_Train_valid=labels_rand(1:floor(0.85*size(Data_original,1)),:);
Data_test=Data_rand_indices(floor(0.85*size(Data_original,1))+1:end,:);
labels_test=labels_rand(floor(0.85*size(Data_original,1))+1:end,:);

%% finding best lambda
% Comment this code if you want to train and test using the optimal lambda
% you already got from this commented section
% iterations=100; % iterations for gradient descent convergence
% lambda=10.^[-2 -1 0 1 2];
% k=5; %k-fold
% Size_per_fold=floor(size(Data_Train_valid,1)/k);
% for cross_valid=1:k
%     
% Valid_indices=1+(cross_valid-1)*Size_per_fold:cross_valid*Size_per_fold;
%         if cross_valid==k & mod(size(Data_Train_valid,1),k)~=0
%             Valid_indices=[Valid_indices size(Data_Train_valid,1)];
%         end
%         Data_validate=Data_Train_valid(Valid_indices,:);
%         labels_validate=labels_Train_valid(Valid_indices,:);
%         Data_Train=Data_Train_valid;
%         labels_Train=labels_Train_valid;
%         Data_Train(Valid_indices,:)=[];
%         labels_Train(Valid_indices,:)=[];
%        
% 
% 
% if OutputNodes>=3
% [~, labels_test_numbers]=max(labels_test,[],2);
% [~, labels_train_numbers]=max(labels_Train,[],2);
% [~, labels_validate_numbers]=max(labels_validate,[],2);
% [~,labels_Train_valid_numbers]=max(labels_Train_valid,[],2);
% end
% 
% 
% 
% [cost_vec,Weights_train,predicted_train,output_train]=Train_ANN_is(lambda(k),iterations,Data_Train,OutputNodes,W,S,Sh,L,alpha,bias,labels_Train,S_vec);
% [Predicted_classes,output_test]=Test_ANN_is(Data_validate,OutputNodes,Weights_train,S,L,bias,S_vec,Sh);
% 
% Weights_crossV(:,:,cross_valid)=Weights_train;
% if OutputNodes>=3
% CCR_validate(cross_valid)=trace(confusionmat(Predicted_classes,labels_validate_numbers))/(length(labels_validate_numbers))
% else
% output_validate_rounded=round(output_test);
% CCR_validate(cross_valid)=trace(confusionmat(output_validate_rounded,labels_validate))/(length(labels_validate))
% end
% 
% 
% end
% %% use the best lambda for all the data now
% max_k=find(CCR_validate==max(CCR_validate));
% if length(max_k)>1
%     max_k=max_k(1);
% end
% lambda_opt=lambda(max_k);
% 

%% now that you know best lambda, only run this part. 
%No need to go through crossvalidation again as it takes time
lambda_opt=1;
iterations2=10000;

if OutputNodes>=3
[~, labels_test_numbers]=max(labels_test,[],2);
[~,labels_Train_valid_numbers]=max(labels_Train_valid,[],2);
end
%[cost_vec,Weights_test,predicted_train,output_train]=Train_ANN_is(lambda(max_k),iterations2,Data_Train_valid,OutputNodes,W,S,Sh,L,alpha,bias,labels_Train_valid,S_vec);
[cost_vec,Weights_test,predicted_train,output_train]=Train_ANN(lambda_opt,iterations2,Data_Train_valid,OutputNodes,W,S,Sh,L,alpha,bias,labels_Train_valid,S_vec);

%% CCR for chunk of valid and train
[Predicted_classes_train,output_total_train]=Test_ANN(Data_Train_valid,OutputNodes,Weights_test,S,L,bias,S_vec,Sh);
%[Predicted_classes_train,output_test_train]=Test_ANN_is(Data_Train_valid,OutputNodes,Weights_test,S,L,bias,S_vec,Sh);

if OutputNodes>=3
 
    CCR_train_total=trace(confusionmat(Predicted_classes_train,labels_Train_valid_numbers))/(length(labels_Train_valid_numbers))
  
else
output_train_total_rounded=round(output_total_train);
CCR_train_total=trace(confusionmat(output_train_total_rounded,labels_Train_valid))/(length(labels_Train_valid))
end

%% testing
[Predicted_classes,output_test]=Test_ANN(Data_test,OutputNodes,Weights_test,S,L,bias,S_vec,Sh);

if OutputNodes>=3
 
    CCR_test=trace(confusionmat(Predicted_classes,labels_test_numbers))/(length(labels_test_numbers))
  
else
output_test_rounded=round(output_test);
CCR_test=trace(confusionmat(output_test_rounded,labels_test))/(length(labels_test))
end


