% RNN_test
clear all; close all; clc
load('RNN1Epo1K.mat')
load('DenoisedPAS.mat')
load('RNNcv.mat')
wieght=CV1.WT;
ktrain=crosVal(1).KTrain;
ktest=crosVal(1).KTest;
ktest1=crosVal(1).KTest1;
ktest2=crosVal(1).KTest2;
inputNum                     =5;% offer, std, contextNeural + context[01 /1 0]
hiddenNum                    =10;%
outputNum                    =1;
e                            =0.005;% learning rate: smaller is better
wtConstr=1/70;

%% plot log likelihood
size(CV1.logL)
mLL=mean(CV1.logL,1)
%%
plot(smooth(mLL,20))

%% test 1 performance without modifying input
for v=1:length(ktest)
    
    k=ktest(v);
    
    for j=1:30 % 30 conditions
        
        output_desired=ActState(k).Ycho(j);
        for t=1:10 % 10 time points
            
            if t<2
                % compute the middle hidden which took input from previous
                % hidden layer and the time point of input
                
                input_temp=[ActState(k).Xoff(j,t),...
                    ActState(k).Xstd(j,t),...
                    ActState(k).Xcxt(j,t),...
                    ActState(k).xcxt(j,:)];
                
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(1,x)=logistic(h_temp(1,x));
                end
                clear x;
            else
                input_temp=[ActState(k).Xoff(j,t),...
                    ActState(k).Xstd(j,t),...
                    ActState(k).Xcxt(j,t),...
                    ActState(k).xcxt(j,:)];
                
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
        Test3Output.ActState(v).condition(j).OUT_desired=output_desired;
        
        
    end% end of 30 conditions in each K
    
    disp('End of set K of 30 conditions, K= ')
    disp(v)
    
end  % end of Ktrain sets of activation states


clear x y z i j;

% save('RNN_Test3.mat','Test3Output');

%% test 1 performance without modifying input
% analysis
y_out=[];
y_tar=[];
for i=1:length(ktest)
    for j=1:30
        y_out=[y_out; Test3Output.ActState(i).condition(j).OUT];
        y_tar=[y_tar; Test3Output.ActState(i).condition(j).OUT_desired];
    end
end

length(y_out)
length(y_tar)

[r, pval]=corrcoef(y_tar, y_out)

my=mean(y_out);
y=y_out;
y(y_out>my)=1;
y(y_out<my)=0;
correct=y==y_tar;
correct_rate=mean(correct);
%% test 2 substituting neural input with tanh(randNum)

for v=1:length(ktest1)
    k=ktest1(v);
    
    for j=1:30 % 30 conditions
        order=randperm(30);
        output_desired=ActState(k).Ycho(j);
        for t=1:10 % 10 time points
            
            %compute hidden unit
            noise_Xcxt = tanh(-1 + (1+1)*rand);
            
            if t<2
                % compute the middle hidden which took input from previous
                % hidden layer and the time point of input
                
                input_temp=[ActState(k).Xoff(j,t),...
                    ActState(k).Xstd(j,t),...
                    noise_Xcxt,...
                    ActState(k).xcxt(j,:)];
                
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(1,x)=logistic(h_temp(1,x));
                end
                clear x;
            else
                input_temp=[ActState(k).Xoff(j,t),...
                    ActState(k).Xstd(j,t),...
                    noise_Xcxt,...
                    ActState(k).xcxt(j,:)];
                
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
        
        CV1test(1).set(v).cdtion(j).OUT=output_temp;
        CV1test(1).set(v).cdtion(j).OUT_desired=output_desired;
        
        
    end% end of 30 conditions in each K
    
    disp('End of set K of 30 conditions, K= ')
    disp(v)
    
end  % end of Ktrain sets of activation states


clear x y z i j;

%% analyze test 2

y_out=[];
y_tar=[];
for i=1:length(ktest1)
    for j=1:30
        y_out=[y_out; CV1test(1).set(v).cdtion(j).OUT];
        y_tar=[y_tar; CV1test(1).set(v).cdtion(j).OUT_desired];
    end
end

length(y_out)
length(y_tar)

[r, pval]=corrcoef(y_tar, y_out)
%% test 3 substituting instructional context input with tanh(randNum)

for v=1:length(ktest2)
    k=ktest2(v);
    
    for j=1:30 % 30 conditions
        order=randperm(30);
        output_desired=ActState(k).Ycho(j);
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
                
                input_temp=[ActState(k).Xoff(j,t),...
                    ActState(k).Xstd(j,t),...
                    ActState(k).Xcxt(j,t),...
                    noise_cxt];
                
                h_temp(t,:)=zeros(1,hiddenNum);
                h_temp(t,end)=1; % biased weights
                for x=1:hiddenNum-1
                    h_temp(1,x)=sum(input_temp.*wieght(1).W(:,x)');
                    h_temp(1,x)=logistic(h_temp(1,x));
                end
                clear x;
            else
                input_temp=[ActState(k).Xoff(j,t),...
                    ActState(k).Xstd(j,t),...
                    ActState(k).Xcxt(j,t),...
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
        
        CV1test(2).set(v).cdtion(j).OUT=output_temp;
        CV1test(2).set(v).cdtion(j).OUT_desired=ActState(k).Ycho(j);
        
        
    end% end of 30 conditions in each K
    
    disp('End of set K of 30 conditions, K= ')
    disp(v)
    
end  % end of Ktrain sets of activation states


clear x y z i j;

%% analyze test 3

y_out=[];
y_tar=[];
for i=1:length(ktest2)
    for j=1:30
        y_out=[y_out; CV1test(2).set(v).cdtion(j).OUT];
        y_tar=[y_tar; CV1test(2).set(v).cdtion(j).OUT_desired];
    end
end

length(y_out)
length(y_tar)

[r, pval]=corrcoef(y_tar, y_out)

