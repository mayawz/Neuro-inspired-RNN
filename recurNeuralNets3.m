%recurNeuralNets3

%% input using

clear all; close all; clc

load('orthogonalizedAxes.mat')
load('DenoisedPopuActiState.mat')

% j=1; k=1;
%%
% input=[1 0 0 0 1;...
%     0 1 0 0 1;...
%     0 0 1 0 1;...
%     0 0 0 1 1 ];
%
% output_ystar=[1 0 0 0;...
%     0 1 0 0;...
%     0 0 1 0;...
%     0 0 0 1];

inputNum                     =5;% offer, std, contextNeural + context[01 /1 0]
hiddenNum                    =25;%
outputNum                    =1;
e                            =0.005;% learning rate: smaller is better
% wtConstr=1/(30*70);
wtConstr=1/70;
epochs                       =1000;

%hidden and output units use the softmax activation function
wieght(1).W=rand(inputNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 1 units
wieght(2).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 2 units
wieght(3).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 3 units
wieght(4).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 4 units
wieght(5).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 5 units
wieght(6).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 6 units
wieght(7).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 7 units
wieght(8).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 8 units
wieght(9).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01;  %weight between input units to hidden 9 units
wieght(10).W=rand(hiddenNum,hiddenNum-1)*0.02-0.01; %weight between hidden 10 units to output units
wieght(11).W=rand(hiddenNum,outputNum)*0.02-0.01;  %weight between hidden 10 units to output units
temp=randsample(70,70);
ktrain=temp(1:40);
ktest1=temp(41:55);
ktest2=temp(56:70);

%%
% logL=zeros(1,epochs);
%do training
for i=1:epochs
    
    for v=1:length(ktrain)
        k=temp(v);
        
        for j=1:30 % 30 conditions
            order=randperm(30);
            output_desired=ActState(k).Ycho(order(j));
            for t=1:10 % 10 time points
                
                
                % ######### first, forward pass #########
                % ######### compute value for every note #########
                
                %compute hidden unit
                if t<2
                    % compute the middle hidden which took input from previous
                    % hidden layer and the time point of input
                    input_temp=[ActState(k).Xoff(order(j),t),...
                        ActState(k).Xstd(order(j),t),...
                        ActState(k).Xcxt(order(j),t),...
                        ActState(k).xcxt(order(j),:)];
                    
                    h_temp(t,:)=zeros(1,hiddenNum);
                    h_temp(t,end)=1; % biased weights
                    for x=1:hiddenNum-1
                        h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                        h_temp(1,x)=logistic(h_temp(1,x));
                    end
                    clear x;
                else
                    input_temp=[ActState(k).Xoff(order(j),t),...
                        ActState(k).Xstd(order(j),t),...
                        ActState(k).Xcxt(order(j),t),...
                        ActState(k).xcxt(order(j),:)];
                    
                    hidden_temp=zeros(1,hiddenNum);
                    hidden_temp(end)=1;
                    h_temp(t,:)=zeros(1,hiddenNum);
                    h_temp(t,end)=1; % biased weights
                    for x=1:hiddenNum-1
                        hidden_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                        h_temp(t,x)=sum(h_temp(t-1,:).*wieght(t).W(:,x)');
                        h_temp(t,x)=h_temp(t,x)+hidden_temp(1,x);
                        h_temp(t,x)=logistic(h_temp(t,x));
                    end
                    
                    clear x;
                    
                end
                
            end % end of 10 time-point input forward passing
            
            %compute output unit
            output_temp=zeros(1,outputNum);
            
            for x=1:outputNum
                output_temp(1,x)=sum(h_temp(10,:).*wieght(11).W(:,x)');
            end
            
            output_temp(1,:)=logistic(output_temp(1,:));
            
            clear x t;
            
            RNNOutput(i).ActState(v).condition(j).OUT=output_temp;
            RNNOutput(i).ActState(v).condition(j).OUT_desired=ActState(k).Ycho(order(j));
            y_tar=ActState(k).Ycho(order(j));
            y_out=output_temp;
            RNNOutput(i).ActState(v).condition(j).logL=y_tar*log(y_out)+(1-y_tar)*log(1-y_out);
            % ######### second, back propagation #########
            % ######### derived weight #########
            
            % compute and update Weight between last hidden unit and output
            
            Error(1:25,1:11)=0;
            deltaWt=[];
            Err=[];
            for x=1:hiddenNum
                for y=1:outputNum
                    % delta=0;
                    Err(1,end+1)=(output_desired/output_temp(1,y)-...
                        (1-output_desired)/(1-output_temp(1,y)))*output_temp(1,y)...
                        *(1-output_temp(1,y));
                    
                    deltaWt(x,y)=Err(end)*h_temp(10,x);
                    
                    wieght(11).W(x,y)=wieght(11).W(x,y)+e * wtConstr * deltaWt(x,y);
                    % clear delta;
                end
            end
            Error(:,11)=Err;
            
            clear x y z;
            
            %compute and update Weight between hidden layer 9 and 10
            deltaWt=[];
            Err=[];
            for x=1:hiddenNum % layer 9
                for y=1:hiddenNum-1 % layer 10
                    % error from output and target
                    err=0;
                    for z=1:outputNum
                        err=err+(output_desired-output_temp(1,z))*wieght(11).W(y,z);
                    end
                    % error at layer 10
                    Err(x,y)=err*h_temp(10,y)*(1-h_temp(10,y));
                    % weight b/w layer 9 and 10
                    deltaWt(x,y)=Err(x,y)*h_temp(9,x);
                    wieght(10).W(x,y)=wieght(10).W(x,y)+e * wtConstr * deltaWt(x,y); %layer
                    clear err;
                end
            end
            Error(:,10)=sum(Err,2);
            
            % compute and update Weight between 9:2 hidden layers
            for t=9:-1:2
                deltaWt=[];
                Err=[];
                for x=1:hiddenNum % layer t-1
                    for y=1:hiddenNum-1 % layer t
                        err=0;
                        for z=1:hiddenNum-1
                            err=err+Error(z,t+1)*wieght(t+1).W(y,z);
                        end
                        % error at layer 9
                        Err(x,y)=err*h_temp(t,y)*(1-h_temp(t,y));
                        % weight b/w layer 9 and 10
                        deltaWt(x,y)=Err(x,y)*h_temp(t-1,x);
                        wieght(t).W(x,y)=wieght(t).W(x,y)+e * wtConstr * deltaWt(x,y); %layer
                        clear err;
                    end
                end
                Error(:,t)=sum(Err,2);
                clear error;
            end
            
            %compute and update Weight between input unit and hidden unit
            deltaWt=[];
            Err=[];
            for x=1:inputNum
                for y=1:hiddenNum-1
                    err=0;
                    %  hwt=1/10;
                    for z=1:hiddenNum-1
                        err=err+Error(z,2)*wieght(2).W(y,z)+...
                            Error(z,3)*wieght(3).W(y,z)+...
                            Error(z,4)*wieght(4).W(y,z)+...
                            Error(z,5)*wieght(5).W(y,z)+...
                            Error(z,6)*wieght(6).W(y,z)+...
                            Error(z,7)*wieght(7).W(y,z)+...
                            Error(z,8)*wieght(8).W(y,z)+...
                            Error(z,9)*wieght(9).W(y,z)+...
                            Error(z,10)*wieght(10).W(y,z);
                        
                    end
                    
                    Err(x,y)=err*h_temp(1,y)*(1-h_temp(1,y));
                    deltaWt(x,y)=Err(x,y)*input_temp(1,x);
                    wieght(1).W(x,y)=wieght(1).W(x,y)+e * wtConstr * deltaWt(x,y); %layer
                    clear err;
                    
                end
            end
            
            %             sum(Err,2)
            
            HidAct(i).set(k).condit(j).H=h_temp;
            
        end% end of 30 conditions in each K
        disp(['Epoch: ' num2str(i) '  Training Instance: '  num2str(v)]);
        
        disp(['logL = ' num2str(RNNOutput(i).ActState(v).condition(j).logL)])
    end  % end of Ktrain sets of activation states
    
end % end of looping through all epochs

clear x y z i j;


% input
WTfull=wieght
% hidden activity
HidAct
% RNN training output
RNNOutput

save('wtHidEpochs.mat','WTfull','HidAct','RNNOutput')

%% test-1 substituting neural input with tanh(randNum)

for v=1:length(ktest1)
    k=temp(v);
    
    for j=1:30 % 30 conditions
        order=randperm(30);
        output_desired=ActState(k).Ycho(order(j));
        for t=1:10 % 10 time points
            
            %compute hidden unit
            noise_Xcxt = tanh(-1 + (1+1)*rand);
            
            if t<2
                % compute the middle hidden which took input from previous
                % hidden layer and the time point of input
                
                input_temp=[ActState(k).Xoff(order(j),t),...
                    ActState(k).Xstd(order(j),t),...
                    noise_Xcxt,...
                    ActState(k).xcxt(order(j),:)];
                
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(1,x)=logistic(h_temp(1,x));
                end
                clear x;
            else
                input_temp=[ActState(k).Xoff(order(j),t),...
                    ActState(k).Xstd(order(j),t),...
                    noise_Xcxt,...
                    ActState(k).xcxt(order(j),:)];
                
                hidden_temp=zeros(1,hiddenNum);
                hidden_temp(end)=1;
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    hidden_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(t,x)=sum(h_temp(t-1,:).*wieght(t).W(:,x)');
                    h_temp(t,x)=h_temp(t,x)+hidden_temp(1,x);
                    h_temp(t,x)=logistic(h_temp(t,x));
                end
                
                clear x;
                
            end
            
        end % end of 10 time-point input forward passing
        
        %compute output unit
        output_temp=zeros(1,outputNum);
        
        for x=1:outputNum
            output_temp(1,x)=sum(h_temp(10,:).*wieght(11).W(:,x)');
        end
        
        output_temp(1,:)=logistic(output_temp(1,:));
        
        clear x t;
        
        Test1Output.ActState(v).condition(j).OUT=output_temp;
        Test1Output.ActState(v).condition(j).OUT_desired=ActState(k).Ycho(order(j));
        
        
    end% end of 30 conditions in each K
    
    disp('End of set K of 30 conditions, K= ')
    disp(v)
    
end  % end of Ktrain sets of activation states


clear x y z i j;

save('RNN_Test1.mat','Test1Output');

%% analyze test-1

y_out=[];
y_tar=[];
for i=1:15
    for j=1:30
        y_out=[y_out; Test1Output.ActState(i).condition(j).OUT];
        y_tar=[y_tar; Test1Output.ActState(i).condition(j).OUT_desired];
    end
end

length(y_out)
length(y_tar)

[r, pval]=corrcoef(y_tar, y_out)
%% test-2 substituting instructional context input with tanh(randNum)

for v=1:length(ktest1)
    k=temp(v);
    
    for j=1:30 % 30 conditions
        order=randperm(30);
        output_desired=ActState(k).Ycho(order(j));
        for t=1:10 % 10 time points
            
            %compute hidden unit
            tempA=rand;
            if tempA >=0.5
                noise_cxt = [1 0];
            else
                noise_cxt = [0 1];
            end
            
            if t<2
                % compute the middle hidden which took input from previous
                % hidden layer and the time point of input
                
                input_temp=[ActState(k).Xoff(order(j),t),...
                    ActState(k).Xstd(order(j),t),...
                    ActState(k).Xcxt(order(j),t),...
                    noise_cxt];
                
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(1,x)=logistic(h_temp(1,x));
                end
                clear x;
            else
                input_temp=[ActState(k).Xoff(order(j),t),...
                    ActState(k).Xstd(order(j),t),...
                    ActState(k).Xcxt(order(j),t),...
                    noise_cxt];
                
                hidden_temp=zeros(1,hiddenNum);
                hidden_temp(end)=1;
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    hidden_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(t,x)=sum(h_temp(t-1,:).*wieght(t).W(:,x)');
                    h_temp(t,x)=h_temp(t,x)+hidden_temp(1,x);
                    h_temp(t,x)=logistic(h_temp(t,x));
                end
                
                clear x;
                
            end
            
        end % end of 10 time-point input forward passing
        
        %compute output unit
        output_temp=zeros(1,outputNum);
        
        for x=1:outputNum
            output_temp(1,x)=sum(h_temp(10,:).*wieght(11).W(:,x)');
        end
        
        output_temp(1,:)=logistic(output_temp(1,:));
        
        clear x t;
        
        Test2Output.ActState(v).condition(j).OUT=output_temp;
        Test2Output.ActState(v).condition(j).OUT_desired=ActState(k).Ycho(order(j));
        
        
    end% end of 30 conditions in each K
    
    disp('End of set K of 30 conditions, K= ')
    disp(v)
    
end  % end of Ktrain sets of activation states


clear x y z i j;

save('RNN_Test2.mat','Test2Output');

%% analyze test-2

y_out=[];
y_tar=[];
for i=1:15
    for j=1:30
        y_out=[y_out; Test2Output.ActState(i).condition(j).OUT];
        y_tar=[y_tar; Test2Output.ActState(i).condition(j).OUT_desired];
    end
end

length(y_out)
length(y_tar)

[r, pval]=corrcoef(y_tar, y_out)

%% test 3 performance without modifying input
tempB=41:70;
clear Test3Output

for v=1:30 % v=1:70
    
    k=tempB(v);% k=temp(v);
    
    for j=1:30 % 30 conditions
        order=randperm(30);
        output_desired=ActState(k).Ycho(order(j));
        for t=1:10 % 10 time points
 
            if t<2
                % compute the middle hidden which took input from previous
                % hidden layer and the time point of input
                
                input_temp=[ActState(k).Xoff(order(j),t),...
                    ActState(k).Xstd(order(j),t),...
                    ActState(k).Xcxt(order(j),t),...
                    ActState(k).xcxt(order(j),:)];
                
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(1,x)=logistic(h_temp(1,x));
                end
                clear x;
            else
                input_temp=[ActState(k).Xoff(order(j),t),...
                    ActState(k).Xstd(order(j),t),...
                    ActState(k).Xcxt(order(j),t),...
                    ActState(k).xcxt(order(j),:)];
                
                hidden_temp=zeros(1,hiddenNum);
                hidden_temp(end)=1;
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    hidden_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(t,x)=sum(h_temp(t-1,:).*wieght(t).W(:,x)');
                    h_temp(t,x)=h_temp(t,x)+hidden_temp(1,x);
                    h_temp(t,x)=logistic(h_temp(t,x));
                end
                
                clear x;
                
            end
            
        end % end of 10 time-point input forward passing
        
        %compute output unit
        output_temp=zeros(1,outputNum);
        
        for x=1:outputNum
            output_temp(1,x)=sum(h_temp(10,:).*wieght(11).W(:,x)');
        end
        
        output_temp(1,:)=logistic(output_temp(1,:));
        
        clear x t;
        
        Test3Output.ActState(v).condition(j).OUT=output_temp;
        Test3Output.ActState(v).condition(j).OUT_desired=ActState(k).Ycho(order(j));
        
        
    end% end of 30 conditions in each K
    
    disp('End of set K of 30 conditions, K= ')
    disp(v)
    
end  % end of Ktrain sets of activation states


clear x y z i j;

% save('RNN_Test3.mat','Test3Output');

%% test 3 performance without modifying input
% analysis
y_out=[];
y_tar=[];
for i=1:30
    for j=1:30
        y_out=[y_out; Test3Output.ActState(i).condition(j).OUT];
        y_tar=[y_tar; Test3Output.ActState(i).condition(j).OUT_desired];
    end
end

length(y_out)
length(y_tar)

[r, pval]=corrcoef(y_tar, y_out)

% save('MayaRNNoutput2.mat')


