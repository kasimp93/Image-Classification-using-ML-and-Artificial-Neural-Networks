function [Predictions,output]=Test_ANN(TestData,OutputNodes,W,S,L,bias,S_vec,Sh)

input=zeros(L,S+1);
output=zeros(size(TestData,1),OutputNodes);



   
for n=1:size(TestData,1)
    %             for n=1:10
    %---------------------Forward Propagation--------------------------
 %---------------------Forward Propagation--------------------------
        input(1,:)=[bias TestData(n,:)];
        for l=2:L
            if l==L
                output(n,:)=logsig(W(end-OutputNodes+1:end,1:Sh+1)*input(L-1,1:S_vec(L-1)+1)')';
            else
                input(l,1:S_vec(l)+1)=[bias logsig(W(sum(S_vec(2:l-1))+1:sum(S_vec(2:l)),1:S_vec(l-1)+1)*input(l-1,1:S_vec(l-1)+1)')'];
            end
        end
   
end
if OutputNodes>2
[~, Predictions]=max(output,[],2);
else
    Predictions=round(output);
end
%[a Predictions]=max(output,[],2);
