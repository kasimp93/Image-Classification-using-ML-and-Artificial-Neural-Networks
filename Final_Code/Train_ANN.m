function [cost_vec,Weights,predicted_train,output]=Train_ANN_is(lambda,iterations,Data,OutputNodes,W,S,Sh,L,alpha,bias,labels,S_vec)


Der=ones(size(W));
counter=2;
m=size(Data,1);
input=zeros(L,S+1);
output=zeros(size(Data,1),OutputNodes);

for ind2=1:iterations
    waitbar(ind2/iterations)%max(max(Der))>accuracy
    max(max(Der))
    delta=zeros(S+1,L-1);
    Delta=zeros(size(W));
    cost=0;
    counter=counter+1;
    for n=1:size(Data,1)
        
        %---------------------Forward Propagation--------------------------
        input(1,:)=[bias Data(n,:)];
        for l=2:L
            if l==L
                output(n,:)=logsig(W(end-OutputNodes+1:end,1:Sh+1)*input(L-1,1:S_vec(L-1)+1)')';
            else
                input(l,1:S_vec(l)+1)=[bias logsig(W(sum(S_vec(2:l-1))+1:sum(S_vec(2:l)),1:S_vec(l-1)+1)*input(l-1,1:S_vec(l-1)+1)')'];
            end
        end
        %-------------------------Compute cost function--------------------
       
        cost=cost-sum(log(output(n,:)).*labels(n,:)+log(1-output(n,:)).*(1-labels(n,:)))/m;
        %------------------------Back Propagation--------------------------
        %delta for the last layer L
        deltaL= output(n,:)'-labels(n,:)';
        Delta(end-OutputNodes+1:end,1:Sh+1)=  Delta(end-OutputNodes+1:end,1:Sh+1)+(deltaL*input(L-1,1:S_vec(L-1)+1));
        
        for back=L-1:-1:2
            g=input(back,1:S_vec(back)+1).*(1-input(back,1:S_vec(back)+1));
            if back==L-1
                delta(1:S_vec(back)+1,back)=(W(end-OutputNodes+1:end,1:Sh+1)'*deltaL).*g';
            else
                delta(1:S_vec(back)+1,back)=(W(sum(S_vec(2:back))+1:sum(S_vec(2:back+1)),1:S_vec(back+1)+1)'*delta(2:S_vec(back+1)+1,back+1)).*g';
            end
            Delta(sum(S_vec(2:back-1))+1:sum(S_vec(2:back)),1:S_vec(back-1)+1)=Delta(sum(S_vec(2:back-1))+1:sum(S_vec(2:back)),1:S_vec(back-1)+1)+delta(2:S_vec(back)+1,back)*input(back-1,1:S_vec(back-1)+1);
        end
    end
    
    %------------------------Compute Dervative-----------------------------
    Der=Delta/m;
    Der(:,2:end)=Der(:,2:end)+lambda/m*W(:,2:end); %add regularization term
%     [grad_approx]=gradient_check2(Data,bias,W,S,L,eps,OutputNodes,labels,lambda,Sh,S_vec)
    cost=cost+lambda/2/m*norm(W(:,2:end),'fro')^2  %before gradient descent step
    cost_vec(counter)=cost;
    %-----------------------------Gradient Descent-----------------------------
    W_old=W;
    W=W_old-alpha*Der;
    
end
Weights=W;
if OutputNodes>2
[~, predicted_train]=max(output,[],2);
else
    predicted_train=round(output);
end
%[a predicted_train]=max(output,[],2);