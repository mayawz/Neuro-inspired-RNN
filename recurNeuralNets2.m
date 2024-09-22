% recurNeuralNets2
% this program organize multiple trials of pupulation response to each
% condition for training and testing of the network.

% this one gives the Population Activation States (PAS) with all
% context*size
% separated

%% response matrix
% state space response matrix only contains 500:750 bins before choice is
% made
close all; clear all; clc
load('SSE13.6040.mat');
cellN=length(data);

%% first, calculate min trial num of each condition pulling all cells
% initialize the min trials size for each condition
minSZ(1:30,:)=999;
for i=1:cellN
    
    c=data(i);
    psth=c.psth;
    
    % NORMALIZE
    psth=psth-mean(psth(:));
    psth=psth./var(psth(:));
    
    % take the time bins analyzed
    
    psth=psth(:,501:750);
    
    vars=c.vars;
    std=vars(:,2);
    offer=vars(:,3);
    Choice=vars(:,4);
    Unchosen=vars(:,5);
    experience=vars(:,6);
    leftStd=vars(:,7);
    choseLeft=vars(:,8);
    
    exptrials=find(experience);
    desctrials=find(~experience);
    
    % condition: each pair of offer std context
    o1s1e=intersect(intersect(find(std==10),find(offer==1)),exptrials);
    o2s1e=intersect(intersect(find(std==10),find(offer==2)),exptrials);
    o3s1e=intersect(intersect(find(std==10),find(offer==3)),exptrials);
    o4s1e=intersect(intersect(find(std==10),find(offer==4)),exptrials);
    o5s1e=intersect(intersect(find(std==10),find(offer==5)),exptrials);
    
    o1s1d=intersect(intersect(find(std==10),find(offer==1)),desctrials);
    o2s1d=intersect(intersect(find(std==10),find(offer==2)),desctrials);
    o3s1d=intersect(intersect(find(std==10),find(offer==3)),desctrials);
    o4s1d=intersect(intersect(find(std==10),find(offer==4)),desctrials);
    o5s1d=intersect(intersect(find(std==10),find(offer==5)),desctrials);
    
    o1s2e=intersect(intersect(find(std==20),find(offer==1)),exptrials);
    o2s2e=intersect(intersect(find(std==20),find(offer==2)),exptrials);
    o3s2e=intersect(intersect(find(std==20),find(offer==3)),exptrials);
    o4s2e=intersect(intersect(find(std==20),find(offer==4)),exptrials);
    o5s2e=intersect(intersect(find(std==20),find(offer==5)),exptrials);
    
    o1s2d=intersect(intersect(find(std==20),find(offer==1)),desctrials);
    o2s2d=intersect(intersect(find(std==20),find(offer==2)),desctrials);
    o3s2d=intersect(intersect(find(std==20),find(offer==3)),desctrials);
    o4s2d=intersect(intersect(find(std==20),find(offer==4)),desctrials);
    o5s2d=intersect(intersect(find(std==20),find(offer==5)),desctrials);
    
    o1s3e=intersect(intersect(find(std==30),find(offer==1)),exptrials);
    o2s3e=intersect(intersect(find(std==30),find(offer==2)),exptrials);
    o3s3e=intersect(intersect(find(std==30),find(offer==3)),exptrials);
    o4s3e=intersect(intersect(find(std==30),find(offer==4)),exptrials);
    o5s3e=intersect(intersect(find(std==30),find(offer==5)),exptrials);
    
    o1s3d=intersect(intersect(find(std==30),find(offer==1)),desctrials);
    o2s3d=intersect(intersect(find(std==30),find(offer==2)),desctrials);
    o3s3d=intersect(intersect(find(std==30),find(offer==3)),desctrials);
    o4s3d=intersect(intersect(find(std==30),find(offer==4)),desctrials);
    o5s3d=intersect(intersect(find(std==30),find(offer==5)),desctrials);
    
    
    trlLen(i,:)=[length(o1s1d) length(o2s1d) length(o3s1d) length(o4s1d) length(o5s1d) ...
                length(o1s2d) length(o2s2d) length(o3s2d) length(o4s2d) length(o5s2d) ...
                length(o1s3d) length(o2s3d) length(o3s3d) length(o4s3d) length(o5s3d) ...
                length(o1s1e) length(o2s1e) length(o3s1e) length(o4s1e) length(o5s1e) ...
                length(o1s2e) length(o2s2e) length(o3s2e) length(o4s2e) length(o5s2e) ...
                length(o1s3e) length(o2s3e) length(o3s3e) length(o4s3e) length(o5s3e)];
    
    cntr(i,1)
end % end of
% min across rows --> get columns for each condition
min(trlLen,[],1)
% min(min(trlLen,[],1))=4
% median(median(trlLen,1))=14 
% mean(mean(trlLen,1))=15 
% length(combnk(1:15,4))= 1365 
% length(combnk(1:14,4))= 1001 
%% since the minSZ=4 calculate population response with 4 trials each time
% but resample 500 times
% randsample(X,4)

for k=1:500
    for i=1:cellN
        
        c=data(i);
        psth=c.psth;
        
        % NORMALIZE
        psth=psth-mean(psth(:));
        psth=psth./var(psth(:));
        
        % take the time bins analyzed
        
        psth=psth(:,501:750);
        
        vars=c.vars;
        std=vars(:,2);
        offer=vars(:,3);
        Choice=vars(:,4);
        Unchosen=vars(:,5);
        experience=vars(:,6);
        leftStd=vars(:,7);
        choseLeft=vars(:,8);
        
        exptrials=find(experience);
        desctrials=find(~experience);
        
        % chose std = -1; offer =1
        choOpt=Choice;
        choOpt(Choice>6)=-1;
        choOpt(Choice<6)=1;
        
        % condition: each pair of offer std context
        o1s1e=intersect(intersect(find(std==10),find(offer==1)),exptrials);
        o2s1e=intersect(intersect(find(std==10),find(offer==2)),exptrials);
        o3s1e=intersect(intersect(find(std==10),find(offer==3)),exptrials);
        o4s1e=intersect(intersect(find(std==10),find(offer==4)),exptrials);
        o5s1e=intersect(intersect(find(std==10),find(offer==5)),exptrials);
        
        o1s1d=intersect(intersect(find(std==10),find(offer==1)),desctrials);
        o2s1d=intersect(intersect(find(std==10),find(offer==2)),desctrials);
        o3s1d=intersect(intersect(find(std==10),find(offer==3)),desctrials);
        o4s1d=intersect(intersect(find(std==10),find(offer==4)),desctrials);
        o5s1d=intersect(intersect(find(std==10),find(offer==5)),desctrials);
        
        % cxt: not neural but designated E: [0 1] D: [1 0]
        % logistic: cho = [0 = chose std OR 1 = chose offer]
        choice= [1 0 0 0;... % OD
                 0 1 0 0;... % OE
                 0 0 1 0;... % SD
                 0 0 0 1];   % SE
        
        temp=randsample(o1s1e,4);
        comb(k).cond(1).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(1).cho(i,:)=choice(4,:);
        comb(k).cond(1).cxt(i,:)=[0 1];
        temp=randsample(o2s1e,4);
        comb(k).cond(2).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(2).cho(i,:)=choice(4,:);
        comb(k).cond(2).cxt(i,:)=[0 1];
        temp=randsample(o3s1e,4);
        comb(k).cond(3).mv(i,:)=mean(psth(temp,:));
        if rand>=0.5
            comb(k).cond(3).cho(i,:)=choice(2,:);
        elseif rand<0.5
            comb(k).cond(3).cho(i,:)=choice(4,:);
        end
        comb(k).cond(3).cxt(i,:)=[0 1];
        temp=randsample(o4s1e,4);
        comb(k).cond(4).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(4).cho(i,:)=choice(2,:);
        comb(k).cond(4).cxt(i,:)=[0 1];
        temp=randsample(o5s1e,4);
        comb(k).cond(5).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(5).cho(i,:)=choice(2,:);
        comb(k).cond(5).cxt(i,:)=[0 1];
        
        temp=randsample(o1s1d,4);
        comb(k).cond(6).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(6).cho(i,:)=choice(3,:);
        comb(k).cond(6).cxt(i,:)=[1 0];
        temp=randsample(o2s1d,4);
        comb(k).cond(7).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(7).cho(i,:)=choice(3,:);
        comb(k).cond(7).cxt(i,:)=[1 0];
        temp=randsample(o3s1d,4);
        comb(k).cond(8).mv(i,:)=mean(psth(temp,:));
        if rand>=0.5
            comb(k).cond(8).cho(i,:)=choice(1,:);
        elseif rand<0.5
            comb(k).cond(8).cho(i,:)=choice(3,:);
        end
        comb(k).cond(8).cxt(i,:)=[1 0];
        temp=randsample(o4s1d,4);
        comb(k).cond(9).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(9).cho(i,:)=choice(1,:);
        comb(k).cond(9).cxt(i,:)=[1 0];
        temp=randsample(o5s1d,4);
        comb(k).cond(10).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(10).cho(i,:)=choice(1,:);
        comb(k).cond(10).cxt(i,:)=[1 0];
        
        o1s2e=intersect(intersect(find(std==20),find(offer==1)),exptrials);
        o2s2e=intersect(intersect(find(std==20),find(offer==2)),exptrials);
        o3s2e=intersect(intersect(find(std==20),find(offer==3)),exptrials);
        o4s2e=intersect(intersect(find(std==20),find(offer==4)),exptrials);
        o5s2e=intersect(intersect(find(std==20),find(offer==5)),exptrials);
        
        o1s2d=intersect(intersect(find(std==20),find(offer==1)),desctrials);
        o2s2d=intersect(intersect(find(std==20),find(offer==2)),desctrials);
        o3s2d=intersect(intersect(find(std==20),find(offer==3)),desctrials);
        o4s2d=intersect(intersect(find(std==20),find(offer==4)),desctrials);
        o5s2d=intersect(intersect(find(std==20),find(offer==5)),desctrials);
        
        temp=randsample(o1s2e,4);
        comb(k).cond(11).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(11).cho(i,:)=choice(4,:);
        comb(k).cond(11).cxt(i,:)=[0 1];
        temp=randsample(o2s2e,4);
        comb(k).cond(12).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(12).cho(i,:)=choice(4,:);
        comb(k).cond(12).cxt(i,:)=[0 1];
        temp=randsample(o3s2e,4);
        comb(k).cond(13).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(13).cho(i,:)=choice(4,:);
        comb(k).cond(13).cxt(i,:)=[0 1];
        temp=randsample(o4s2e,4);
        comb(k).cond(14).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(14).cho(i,:)=choice(2,:);
        comb(k).cond(14).cxt(i,:)=[0 1];
        temp=randsample(o5s2e,4);
        comb(k).cond(15).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(15).cho(i,:)=choice(2,:);
        comb(k).cond(15).cxt(i,:)=[0 1];
        
        temp=randsample(o1s2d,4);
        comb(k).cond(16).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(16).cho(i,:)=choice(3,:);
        comb(k).cond(16).cxt(i,:)=[1 0];
        temp=randsample(o2s2d,4);
        comb(k).cond(17).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(17).cho(i,:)=choice(3,:);
        comb(k).cond(17).cxt(i,:)=[1 0];
        temp=randsample(o3s2d,4);
        comb(k).cond(18).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(18).cho(i,:)=choice(3,:);
        comb(k).cond(18).cxt(i,:)=[1 0];
        temp=randsample(o4s2d,4);
        comb(k).cond(19).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(19).cho(i,:)=choice(1,:);
        comb(k).cond(19).cxt(i,:)=[1 0];
        temp=randsample(o5s2d,4);
        comb(k).cond(20).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(20).cho(i,:)=choice(1,:);
        comb(k).cond(20).cxt(i,:)=[1 0];
        
        o1s3e=intersect(intersect(find(std==30),find(offer==1)),exptrials);
        o2s3e=intersect(intersect(find(std==30),find(offer==2)),exptrials);
        o3s3e=intersect(intersect(find(std==30),find(offer==3)),exptrials);
        o4s3e=intersect(intersect(find(std==30),find(offer==4)),exptrials);
        o5s3e=intersect(intersect(find(std==30),find(offer==5)),exptrials);
        
        o1s3d=intersect(intersect(find(std==30),find(offer==1)),desctrials);
        o2s3d=intersect(intersect(find(std==30),find(offer==2)),desctrials);
        o3s3d=intersect(intersect(find(std==30),find(offer==3)),desctrials);
        o4s3d=intersect(intersect(find(std==30),find(offer==4)),desctrials);
        o5s3d=intersect(intersect(find(std==30),find(offer==5)),desctrials);
        
        temp=randsample(o1s3e,4);
        comb(k).cond(21).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(21).cho(i,:)=choice(4,:);
        comb(k).cond(21).cxt(i,:)=[0 1];
        temp=randsample(o2s3e,4);
        comb(k).cond(22).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(22).cho(i,:)=choice(4,:);
        comb(k).cond(22).cxt(i,:)=[0 1];
        temp=randsample(o3s3e,4);
        comb(k).cond(23).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(23).cho(i,:)=choice(4,:);
        comb(k).cond(23).cxt(i,:)=[0 1];
        temp=randsample(o4s3e,4);
        comb(k).cond(24).mv(i,:)=mean(psth(temp,:));
        if rand>=0.5
            comb(k).cond(24).cho(i,:)=choice(2,:);
        elseif rand<0.5
            comb(k).cond(24).cho(i,:)=choice(4,:);
        end
        comb(k).cond(24).cxt(i,:)=[0 1];
        temp=randsample(o5s3e,4);
        comb(k).cond(25).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(25).cho(i,:)=choice(2,:);
        comb(k).cond(25).cxt(i,:)=[0 1];
        
        temp=randsample(o1s3d,4);
        comb(k).cond(26).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(26).cho(i,:)=choice(3,:);
        comb(k).cond(26).cxt(i,:)=[1 0];
        temp=randsample(o2s3d,4);
        comb(k).cond(27).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(27).cho(i,:)=choice(3,:);
        comb(k).cond(27).cxt(i,:)=[1 0];
        temp=randsample(o3s3d,4);
        comb(k).cond(28).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(28).cho(i,:)=choice(3,:);
        comb(k).cond(28).cxt(i,:)=[1 0];
        temp=randsample(o4s3d,4);
        comb(k).cond(29).mv(i,:)=mean(psth(temp,:));
        if rand>=0.5
            comb(k).cond(29).cho(i,:)=choice(1,:);
        elseif rand<0.5
            comb(k).cond(29).cho(i,:)=choice(3,:);
        end
        comb(k).cond(29).cxt(i,:)=[1 0];
        temp=randsample(o5s3d,4);
        comb(k).cond(30).mv(i,:)=mean(psth(temp,:));
        comb(k).cond(30).cho(i,:)=choice(1,:);
        comb(k).cond(30).cxt(i,:)=[1 0];
        
        cntr(i,25)
        
    end % end of cell i
    
    cntr(k,1)
end % end of resample combination k

%% reducing time point to 250ms/ 10 time points
for k=1:500
    for j=1:30
        for t=1:10
            
            cob(k).con(j).M(:,t)=mean(comb(k).cond(j).mv(:,(t-1)*25+1:t*25),2);
            
        end % end of 10 time points
        
    end % end of 30 conditions
    
end % end of k combinations of resampling

cob(1).con(1)
%% de-noise the population response
clear a b 

load('denoiseMatrix')

for k=1:500
    for j=1:30
      for t=1:10
        a=cob(k).con(j).M(:,t);
        b=D(t).d*a;
        deNoised.cob(k).con(j).M(:,t)=b;

      end
    end
end

deNoised.cob(k).con(j)
clear a b 
%%
%   Borth which we refer to as the ?task-related axes? of choice, offer,
%   std, and context. These axes span the same ?regression subspace? as
%   the original regression vectors, but crucially each explains distinct
%   portions of the variance in the responses.

load('orthogonalizedAxes.mat')
for k=1:500
    for j=1:30
        
        TEMP=Borth'* deNoised.cob(k).con(j).M;
        
        PAS(k).frChoice(j,:)=TEMP(1,:);
        PAS(k).frOffer(j,:)=TEMP(2,:);
        PAS(k).frStd(j,:)=TEMP(3,:);
        PAS(k).frCxt(j,:)=TEMP(4,:);
        
        PAS(k).target(j,:)=median(comb(k).cond(j).cho);
        PAS(k).instructCxt(j,:)=mean(comb(k).cond(j).cxt);
        
        clear TEMP
    end
end
%%
% save the population activation state resampled 70 times (for trials to
% average) and for each of 30 combo of context and offer*std size
save('DenoisedPAS.mat', 'PAS')

% The outcomes were set to be the correct one. But in terms of estimating
% network performance, the best should be monkeys 85% correct performance









