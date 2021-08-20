function [W,numerical] = GLLE(logicalLabel,features,C_low_rank,lambda)
[d,n] = size(features);
[l,~] = size(logicalLabel);

[kk,~] = size(C_low_rank);

global   trainFeature;
global   trainLabel;
global   para;
global   CC;

%%%%%%%linear model,kernel method%%%%%%%%
ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
par  = 1*mean(pdist(features)); %parameter of kernel function
H = kernelmatrix(ker, par, features, features);% build the kernel matrix on the labeled samples (N x N)
UnitMatrix = ones(size(features,1),1);
trainFeature = [H,UnitMatrix];
%%%%%%%linear model,kernel method%%%%%%%%

CC = (eye(kk)-C_low_rank')*(eye(kk)-C_low_rank);
trainLabel = logicalLabel;
para = lambda;
item=rand(size(trainFeature,2),size(trainLabel,2));
%save dt.mat trainFeature trainLabel G lambda
[W,fval] = fminlbfgsGLLE(@LEbfgsProcess,item);
numerical = trainFeature*W;


end
