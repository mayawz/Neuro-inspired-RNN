% RNN_CV1

clear all; close all; clc

load('DenoisedPopuActiState.mat')
load('RNNcrosVal.mat')

saveWtEpo=200:50:2500;
inputNum                     =5;% offer, std, contextNeural + context[01 /1 0]
hiddenNum                    =10;%
outputNum                    =1;
e                            =0.005;% learning rate: smaller is better
wtConstr                     =1/100;
epochs                       =1000;

%hidden and output units use the softmax activation function
WT(1).W=rand(inputNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 1 units
WT(2).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 2 units
WT(3).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 3 units
WT(4).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 4 units
WT(5).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 5 units
WT(6).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 6 units
WT(7).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 7 units
WT(8).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 8 units
WT(9).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004;  %weight between input units to hidden 9 units
WT(10).W=rand(hiddenNum,hiddenNum-1)*0.005-0.004; %weight between hidden 10 units to output units
WT(11).W=rand(hiddenNum,outputNum)*0.005-0.004;  %weight between hidden 10 units to output units

ktrain=crosVal(1).KTrain;
ktest=crosVal(1).KTest;
ktest1=crosVal(1).KTest1;
ktest2=crosVal(1).KTest2;

%%
for i=1:epochs
    
    t1=length(ktrain);
    temp=ktrain(randsample(t1,t1));
    
    for v=1:t1
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
                        h_temp(1,x)=sum(input_temp.*WT(1).W(:,x)');
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
                        hidden_temp(1,x)=sum(input_temp.*WT(1).W(:,x)');
                        h_temp(t,x)=sum(h_temp(t-1,:).*WT(t).W(:,x)');
                        h_temp(t,x)=h_temp(t,x)+hidden_temp(1,x);
                        h_temp(t,x)=logistic(h_temp(t,x));
                    end
                    
                    clear x;
                    
                end
                
            end % end of 10 time-point input forward passing
            
            %compute output unit
            output_temp=zeros(1,outputNum);
            
            for x=1:outputNum
                output_temp(1,x)=sum(h_temp(10,:).*WT(11).W(:,x)');
            end
            
            output_temp(1,:)=logistic(output_temp(1,:));
            
            clear x t;
            
            if i > 1500
            CV1.epoc(i).set(v).cdtion(j).trainOUT=output_temp;
            CV1.epoc(i).set(v).cdtion(j).trainTAR=output_desired;
            CV1.epoc(i).set(v).cdtion(j).H=h_temp;
            CV1.epoc(i).set(v).cdtion(j).H1=hidden_temp;
            end
            y_tar=output_desired;
            y_out=output_temp;
            
            ll(j)=y_tar*log(y_out)+(1-y_tar)*log(1-y_out);
   
            % ######### second, back propagation #########
            % ######### derived weight #########
            
            % compute and update Weight between last hidden unit and output
            
            Error(1:10,1:11)=0;
            deltaWt=[];
            Err=[];
            for x=1:hiddenNum
                for y=1:outputNum
                    % delta=0;
                    Err(1,end+1)=(output_desired/output_temp(1,y)-...
                        (1-output_desired)/(1-output_temp(1,y)))*output_temp(1,y)...
                        *(1-output_temp(1,y));
                    
                    deltaWt(x,y)=Err(end)*h_temp(10,x);
                    
                    WT(11).W(x,y)=WT(11).W(x,y)+e * wtConstr * deltaWt(x,y);
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
                        err=err+(output_desired-output_temp(1,z))*WT(11).W(y,z);
                    end
                    % error at layer 10
                    Err(x,y)=err*h_temp(10,y)*(1-h_temp(10,y));
                    % weight b/w layer 9 and 10
                    deltaWt(x,y)=Err(x,y)*h_temp(9,x);
                    WT(10).W(x,y)=WT(10).W(x,y)+e * wtConstr * deltaWt(x,y); %layer
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
                            err=err+Error(z,t+1)*WT(t+1).W(y,z);
                        end
                        % error at layer 9
                        Err(x,y)=err*h_temp(t,y)*(1-h_temp(t,y));
                        % weight b/w layer 9 and 10
                        deltaWt(x,y)=Err(x,y)*h_temp(t-1,x);
                        WT(t).W(x,y)=WT(t).W(x,y)+e * wtConstr * deltaWt(x,y); %layer
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
                        err=err+Error(z,2)*WT(2).W(y,z)+...
                            Error(z,3)*WT(3).W(y,z)+...
                            Error(z,4)*WT(4).W(y,z)+...
                            Error(z,5)*WT(5).W(y,z)+...
                            Error(z,6)*WT(6).W(y,z)+...
                            Error(z,7)*WT(7).W(y,z)+...
                            Error(z,8)*WT(8).W(y,z)+...
                            Error(z,9)*WT(9).W(y,z)+...
                            Error(z,10)*WT(10).W(y,z);
                        
                    end
                    
                    Err(x,y)=err*h_temp(1,y)*(1-h_temp(1,y));
                    deltaWt(x,y)=Err(x,y)*input_temp(1,x);
                    WT(1).W(x,y)=WT(1).W(x,y)+e * wtConstr * deltaWt(x,y); %layer
                    clear err;
                    
                end
            end
            
            %             sum(Err,2)
                        
        end% end of 30 conditions in each set
        
        CV1.logL(v,i)=sum(ll);
        disp('RNN_CV1')
        disp(['Epoch: ' num2str(i) '  Training Instance: '  num2str(v)]);
        disp(['logL = ' num2str(CV1.logL(v,i))])
        clear ll;
        
    end  % end of Ktrain sets of activation states
    
    if ismember(i,saveWtEpo)
        CV1.midWt(i/50).WT=WT;
    end
    
end % end of looping through all epochs

clear x y z i j;


CV1.WT=WT;

save('RNN_CV1.mat','CV1')

