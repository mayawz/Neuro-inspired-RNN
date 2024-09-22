% recurNeuralNets
% part 1 for robbie's computational method class final project

clear all; close all; clc

load('SSE13.6040.mat');
cellN=length(data);

%%
for i=1:cellN
    
    c=data(i);
    psth=c.psth;
    %     psth=psth(:,451:1200);
    psth=psth(:,500:750);
    
    % NORMALIZE
    psth=psth-mean(psth(:));
    psth=psth./var(psth(:));
    
    vars=c.vars;
    std=vars(:,2);
    offer=vars(:,3);
    Choice=vars(:,4);
    Unchosen=vars(:,5);
    experience=vars(:,6);
    leftStd=vars(:,7);
    choseLeft=vars(:,8);
    
    offerSize=[75 100 150 200 250];
    offerJ=offerSize(offer);
    
    stdSize=[150 175 200];
    stdJ=stdSize(std/10);
    
    choOpt=Choice;
    % chose std = -1; offer =1
    choOpt(Choice>6)=-1;
    choOpt(Choice<6)=1;
    
    choice=Choice;
    choice(find(Choice>6))=stdJ(find(Choice>6));
    choice(find(Choice<6))=offerJ(find(Choice<6));
    
    unchosen=Unchosen;
    unchosen(find(Unchosen>6))=stdJ(find(Unchosen>6));
    unchosen(find(Unchosen<6))=offerJ(find(Unchosen<6));
    
    correctTrl=find(choice>=unchosen);
    
    % z score all regressors
    
    zchoOpt=choOpt;
    zofferJ=zscore(offerJ)';
    zstdJ=zscore(stdJ)';
    zcontext=zscore(experience);
    
    %     zchoOpt=zchoOpt(correctTrl); zofferJ=zofferJ(correctTrl);
    %     zstdJ=zstdJ(correctTrl); zcontext=zcontext(correctTrl);
    
    for x=1:10 % 1:720
        subpsth=psth(:,(x-1)*25+1:x*25);
        fr=mean(subpsth,2);
        %         fr=fr(correctTrl,:);
        tbl = table(fr,zchoOpt,zofferJ,zstdJ,zcontext,...
            'VariableNames',{'fr','zchoOpt','zofferJ','zstdJ','zcontext'});
        
        lm = fitlm(tbl,'fr~zchoOpt+zofferJ+zstdJ+zcontext');
        
        b.constant(i,x)=lm.Coefficients.Estimate(1);
        b.choOpt(i,x)=lm.Coefficients.Estimate(2);
        b.off(i,x)=lm.Coefficients.Estimate(3);
        b.std(i,x)=lm.Coefficients.Estimate(4);
        b.zcontext(i,x)=lm.Coefficients.Estimate(5);
        
        p.choOpt(i,x)=lm.Coefficients.pValue(2);
        p.off(i,x)=lm.Coefficients.pValue(3);
        p.std(i,x)=lm.Coefficients.pValue(4);
        p.zcontext(i,x)=lm.Coefficients.pValue(5);
        
        clear ml tbl
        
        cntr(x,10)
        
    end % end of sliding regression
    
    cntr(i,1)
end % end of looping throgh cells

%%
save('RNN_regresR.mat','b','p')

%% response matrix
% state space response matrix only contains 500:750 bins before choice is
% made
close all; clear all; clc
load('SSE13.6040.mat');
cellN=length(data);

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
    
    cond.o1s1e(i,:)=mean(psth(o1s1e,:));
    cond.o2s1e(i,:)=mean(psth(o2s1e,:));
    cond.o3s1e(i,:)=mean(psth(o3s1e,:));
    cond.o4s1e(i,:)=mean(psth(o4s1e,:));
    cond.o5s1e(i,:)=mean(psth(o5s1e,:));
    
    o1s1d=intersect(intersect(find(std==10),find(offer==1)),desctrials);
    o2s1d=intersect(intersect(find(std==10),find(offer==2)),desctrials);
    o3s1d=intersect(intersect(find(std==10),find(offer==3)),desctrials);
    o4s1d=intersect(intersect(find(std==10),find(offer==4)),desctrials);
    o5s1d=intersect(intersect(find(std==10),find(offer==5)),desctrials);
    
    cond.o1s1d(i,:)=mean(psth(o1s1d,:));
    cond.o2s1d(i,:)=mean(psth(o2s1d,:));
    cond.o3s1d(i,:)=mean(psth(o3s1d,:));
    cond.o4s1d(i,:)=mean(psth(o4s1d,:));
    cond.o5s1d(i,:)=mean(psth(o5s1d,:));
    
    o1s2e=intersect(intersect(find(std==20),find(offer==1)),exptrials);
    o2s2e=intersect(intersect(find(std==20),find(offer==2)),exptrials);
    o3s2e=intersect(intersect(find(std==20),find(offer==3)),exptrials);
    o4s2e=intersect(intersect(find(std==20),find(offer==4)),exptrials);
    o5s2e=intersect(intersect(find(std==20),find(offer==5)),exptrials);
    
    cond.o1s2e(i,:)=mean(psth(o1s2e,:));
    cond.o2s2e(i,:)=mean(psth(o2s2e,:));
    cond.o3s2e(i,:)=mean(psth(o3s2e,:));
    cond.o4s2e(i,:)=mean(psth(o4s2e,:));
    cond.o5s2e(i,:)=mean(psth(o5s2e,:));
    
    o1s2d=intersect(intersect(find(std==20),find(offer==1)),desctrials);
    o2s2d=intersect(intersect(find(std==20),find(offer==2)),desctrials);
    o3s2d=intersect(intersect(find(std==20),find(offer==3)),desctrials);
    o4s2d=intersect(intersect(find(std==20),find(offer==4)),desctrials);
    o5s2d=intersect(intersect(find(std==20),find(offer==5)),desctrials);
    
    cond.o1s2d(i,:)=mean(psth(o1s2d,:));
    cond.o2s2d(i,:)=mean(psth(o2s2d,:));
    cond.o3s2d(i,:)=mean(psth(o3s2d,:));
    cond.o4s2d(i,:)=mean(psth(o4s2d,:));
    cond.o5s2d(i,:)=mean(psth(o5s2d,:));
    
    o1s3e=intersect(intersect(find(std==30),find(offer==1)),exptrials);
    o2s3e=intersect(intersect(find(std==30),find(offer==2)),exptrials);
    o3s3e=intersect(intersect(find(std==30),find(offer==3)),exptrials);
    o4s3e=intersect(intersect(find(std==30),find(offer==4)),exptrials);
    o5s3e=intersect(intersect(find(std==30),find(offer==5)),exptrials);
    
    cond.o1s3e(i,:)=mean(psth(o1s3e,:));
    cond.o2s3e(i,:)=mean(psth(o2s3e,:));
    cond.o3s3e(i,:)=mean(psth(o3s3e,:));
    cond.o4s3e(i,:)=mean(psth(o4s3e,:));
    cond.o5s3e(i,:)=mean(psth(o5s3e,:));
    
    o1s3d=intersect(intersect(find(std==30),find(offer==1)),desctrials);
    o2s3d=intersect(intersect(find(std==30),find(offer==2)),desctrials);
    o3s3d=intersect(intersect(find(std==30),find(offer==3)),desctrials);
    o4s3d=intersect(intersect(find(std==30),find(offer==4)),desctrials);
    o5s3d=intersect(intersect(find(std==30),find(offer==5)),desctrials);
    
    cond.o1s3d(i,:)=mean(psth(o1s3d,:));
    cond.o2s3d(i,:)=mean(psth(o2s3d,:));
    cond.o3s3d(i,:)=mean(psth(o3s3d,:));
    cond.o4s3d(i,:)=mean(psth(o4s3d,:));
    cond.o5s3d(i,:)=mean(psth(o5s3d,:));
    
    
    cntr(i,1)
end % end of


%% reducing time point to 250ms/ 10 time points
for k=1:10
    
    con(1).M(:,k)=mean(cond.o1s1e(:,(k-1)*25+1:k*25),2);
    con(2).M(:,k)=mean(cond.o2s1e(:,(k-1)*25+1:k*25),2);
    con(3).M(:,k)=mean(cond.o3s1e(:,(k-1)*25+1:k*25),2);
    con(4).M(:,k)=mean(cond.o4s1e(:,(k-1)*25+1:k*25),2);
    con(5).M(:,k)=mean(cond.o5s1e(:,(k-1)*25+1:k*25),2);
    
    con(6).M(:,k)=mean(cond.o1s1d(:,(k-1)*25+1:k*25),2);
    con(7).M(:,k)=mean(cond.o2s1d(:,(k-1)*25+1:k*25),2);
    con(8).M(:,k)=mean(cond.o3s1d(:,(k-1)*25+1:k*25),2);
    con(9).M(:,k)=mean(cond.o4s1d(:,(k-1)*25+1:k*25),2);
    con(10).M(:,k)=mean(cond.o5s1d(:,(k-1)*25+1:k*25),2);
    
    con(11).M(:,k)=mean(cond.o1s2e(:,(k-1)*25+1:k*25),2);
    con(12).M(:,k)=mean(cond.o2s2e(:,(k-1)*25+1:k*25),2);
    con(13).M(:,k)=mean(cond.o3s2e(:,(k-1)*25+1:k*25),2);
    con(14).M(:,k)=mean(cond.o4s2e(:,(k-1)*25+1:k*25),2);
    con(15).M(:,k)=mean(cond.o5s2e(:,(k-1)*25+1:k*25),2);
    
    con(16).M(:,k)=mean(cond.o1s2d(:,(k-1)*25+1:k*25),2);
    con(17).M(:,k)=mean(cond.o2s2d(:,(k-1)*25+1:k*25),2);
    con(18).M(:,k)=mean(cond.o3s2d(:,(k-1)*25+1:k*25),2);
    con(19).M(:,k)=mean(cond.o4s2d(:,(k-1)*25+1:k*25),2);
    con(20).M(:,k)=mean(cond.o5s2d(:,(k-1)*25+1:k*25),2);
    
    con(21).M(:,k)=mean(cond.o1s3e(:,(k-1)*25+1:k*25),2);
    con(22).M(:,k)=mean(cond.o2s3e(:,(k-1)*25+1:k*25),2);
    con(23).M(:,k)=mean(cond.o3s3e(:,(k-1)*25+1:k*25),2);
    con(24).M(:,k)=mean(cond.o4s3e(:,(k-1)*25+1:k*25),2);
    con(25).M(:,k)=mean(cond.o5s3e(:,(k-1)*25+1:k*25),2);
    
    con(26).M(:,k)=mean(cond.o1s3d(:,(k-1)*25+1:k*25),2);
    con(27).M(:,k)=mean(cond.o2s3d(:,(k-1)*25+1:k*25),2);
    con(28).M(:,k)=mean(cond.o3s3d(:,(k-1)*25+1:k*25),2);
    con(29).M(:,k)=mean(cond.o4s3d(:,(k-1)*25+1:k*25),2);
    con(30).M(:,k)=mean(cond.o5s3d(:,(k-1)*25+1:k*25),2);
    
end

con(1)

%% each time point all condition all cells

% [coeff,score,latent,~,explained] = pca(X) returns the principal component
% coefficients, also known as loadings, for the n-by-p data matrix X. Rows
% of X correspond to observations and columns correspond to variables. The
% coefficient matrix is p-by-p. Each column of coeff contains coefficients
% for one principal component, and the columns are in descending order of
% component variance. By default, pca centers the data and uses the
% singular value decomposition (SVD) algorithm.

% Principal component scores
% are the representations of X in the principal component space. Rows of
% score correspond to observations, and columns correspond to components.

% Xcentered = score*coeff' The new data in Xcentered is the original
% ingredients data centered by subtracting the column means from
% corresponding columns.

% explained ? Percentage of total variance explained

% The easiest way to understand PCA is using eigenvalue decomposition of
% the covariance matrix Sigma:
% Sigma = V*Lambda*V'
% Lambda is the diagonal matrix of eigenvalues. V is an orthonormal matrix
% of coefficients. Orthonormality implies that the 2-norm of every column
% is 1.
% This is what the MATLAB implementation does.
for t=1:10
    for j=1:30
        T(t).m(j,:)= con(j).M(:,t)';
    end
end

% for each time point, size(T(t).m) = 30 conditions * 125 neurons
%% PCA
for t=1:10
    
    [pc,score,latent,~,explained]=pca(T(t).m);
    
    PCAmat(t).pc=pc;
    PCAmat(t).score=score;
    PCAmat(t).eigenVal=latent;
    PCAmat(t).explained=explained;
end

%% plot eigen value
for t=1:10
    figure(t)
    subplot(1,2,1)
    plot(1:length(PCAmat(t).eigenVal),PCAmat(t).eigenVal,'o-');
    hold on
    vline(10)
    hold off
    subplot(1,2,2)
    plot(1:length(PCAmat(t).explained),PCAmat(t).explained,'s-');
    hold on
    vline(10)
    hline(2)
    hold off
    
%     saveas(gcf,['EigenVal_Time' num2str(t)],'epsc')
%     close all
    
end

% by checking the figures, first 10 PC are chosen
%% for each time point t --> a de-noise matrix D(t) using only first 10 PCs

for t=1:10
    temp(1:125,1:125)=0;
    for p=1:10
        temp=temp+PCAmat(t).pc(:,p)*PCAmat(t).pc(:,p)';
    end
    D(t).d=temp;
end

% the de-noise matrix is 125*125

%% Xpca: de-noised population response at each time: condition * Ncells
for t=1:10
    Xpca(t).x=D(t).d*T(t).m';
end

% size(Xpca(t).x) = 125(Ncell)*30(condition) matrix

save('denoiseMatrix','D')
%% regression Subspace
clc; clear b p
Reg=load('RNN_regresR.mat','b','p')


% the fundamental conceptual step: viewing the regression coefficients
% not as properties of individual units, but as the directions in state
% space along which the underlying task variables are represented at the
% level of the population. 

% Each vector, beta per condition per time point, thus corresponds to a
% direction in state space that accounts for variance in the population
% response at that time point, due to variation in task variable  .

%% calculating de-noised regression coefficients
for t=1:10
    Betapca(t).off=D(t).d*Reg.b.off(:,t);
    Betapca(t).std=D(t).d*Reg.b.std(:,t);
    Betapca(t).cnxt=D(t).d*Reg.b.zcontext(:,t);
    Betapca(t).choOpt=D(t).d*Reg.b.choOpt(:,t);
    Betapca(t).constant=D(t).d*Reg.b.constant(:,t);
    
    BetaPca.off(:,t)=D(t).d*Reg.b.off(:,t);
    BetaPca.std(:,t)=D(t).d*Reg.b.std(:,t);
    BetaPca.cnxt(:,t)=D(t).d*Reg.b.zcontext(:,t);
    BetaPca.choOpt(:,t)=D(t).d*Reg.b.choOpt(:,t);
    BetaPca.constant(:,t)=D(t).d*Reg.b.constant(:,t);
end

%%

for t=1:10
     
    no(t)=norm(BetaPca.off(:,t));
    ns(t)=norm(BetaPca.std(:,t));
    nct(t)=norm(BetaPca.cnxt(:,t));
    nch(t)=norm(BetaPca.choOpt(:,t));
    ncs(t)=norm(BetaPca.constant(:,t));
    
end

% [argvalue, argmin] = min(x);
% [argvalue, argmax] = max(x);

[~, offMaxT] = max(no)
[~, stdMaxT] = max(ns)
[~, cnxtMaxT] = max(nct)
[~, choOptMaxT] = max(nch)
[~, constantMaxT] = max(ncs)

betaMax.off=BetaPca.off(:,offMaxT);
betaMax.std=BetaPca.std(:,stdMaxT);
betaMax.cnxt=BetaPca.cnxt(:,cnxtMaxT);
betaMax.choOpt=BetaPca.choOpt(:,choOptMaxT);
betaMax.constant=BetaPca.constant(:,constantMaxT);

Bmax=[betaMax.choOpt betaMax.off betaMax.std betaMax.cnxt];

%% QR decomposition
%   is an orthogonal matrix, and   is an upper triangular matrix. The first
%   four columns of   correspond to the orthogonalized regression vectors, 
%   Borth which we refer to as the ?task-related axes? of choice, offer,
%   std, and context. These axes span the same ?regression subspace? as
%   the original regression vectors, but crucially each explains distinct
%   portions of the variance in the responses.

[Q,R] = qr(Bmax);

Borth=Q(:,1:4);

%%
% save the orthogonalized Axes
save('orthogonalizedAxes.mat','Borth')

%% project the average population responses onto these orthogonal axes

% this is used to reproduce Fig2 in Mante 2013 Nature

% population response in each defined condition in this, I used 30
% conditions

for j=1:30
 popResp(j).c= Borth'* con(j).M;
end

% Interpret the projection of the responses onto the choice axis,
% Borth(:,1), as the integrated relevant evidence
%%

% PLOT THE POPULATION RESPONSE ON EACH AXIS GIVEN PER 
% INTERESTING CONDITION  

plot(popResp(1).c','s-')
% legend('choice','offer','standard','context')
hold on
plot(popResp(6).c','*-')
legend('choice','offer','standard','context','choice','offer','standard','context')

%%


