
% what you need to run this code is the weights for all the features and
% for all 2, 15 classes, the images vector of the extracted data and the corresponding images.
% i have provided the weights for 15 scene combined and gist as examples
% but please make sure to comment the features you are not using, such as 2
% classes test and centrist

%%GIST
%% results 2 class
image=load('X_gist.mat');
image=image.image;
Scene15=image([2 3 4 8],:);
Scene2=image([10 11 12 13 14 15 16 18 19 21 22],:); 


% Weights_test=load('Weights_gist_2_class_hidden200_iter10000_lam1.mat');
% Weights_test=Weights_test.Weights_test;
% S=size(image,2);
% S_vec=[S 200 200 1]; 
% [P,output_test]=Test_ANN(Scene2,1,Weights_test,S,4,1,S_vec,200);
% Predicted2=round(output_test)

Weights_test=load('Weights_gist_15class_10000it_lam1_hidden200.mat');
Weights_test=Weights_test.Weights_test;
S=size(image,2);
S_vec=[S 200 200 15]; 
[P15,output_test]=Test_ANN(Scene15,15,Weights_test,S,4,1,S_vec,200);
P15


%% COMB
%% results 2 class
image=load('X_comb.mat');
image=image.X_comb;
Scene15=image([2 3 4 8],:);
Scene2=image([9 10 11 12 13 14 15 16 18 19 21 22],:); %13 instead of 2


% Weights_test=load('Weights_comb_2class_it1000_lam01_hidden200.mat');
% Weights_test=Weights_test.Weights_test;
% S=size(image,2);
% S_vec=[S 200 200 1]; 
% [P,output_test]=Test_ANN(Scene2,1,Weights_test,S,4,1,S_vec,200);
% Predicted2=round(output_test)

Weights_test=load('Weights_comb_15class_it1000_lam100_hidden200.mat');
Weights_test=Weights_test.Weights_test;
S=size(image,2);
S_vec=[S 200 200 15]; 
[P15,output_test]=Test_ANN(Scene15,15,Weights_test,S,4,1,S_vec,200);

P15
%% CENT
% %% results 2 class
% image=load('X_cent.mat');
% image=image.X_cent;
% Scene15=image([2 3 4 8],:);
% Scene2=image([9 10 11 12 13 14 15 16 18 19 21 22],:); %13 instead of 2
% 
% Weights_test=load('Weights_cent_2class_it1000_lam1_hidden128.mat');
% Weights_test=Weights_test.Weights_test;
% S=size(image,2);
% S_vec=[S 128 128 1]; 
% [P,output_test]=Test_ANN(Scene2,1,Weights_test,S,4,1,S_vec,128);
% Predicted2=round(output_test)
% 
% Weights_test=load('Weights_cent_15class_it1000_lam10_hidden128.mat');
% Weights_test=Weights_test.Weights_test;
% S=size(image,2);
% S_vec=[S 128 128 15]; 
% [P15,output_test]=Test_ANN(Scene15,15,Weights_test,S,4,1,S_vec,128);
% 
% P15
