clc
clear
addpath(genpath(pwd));

dataset={'LDL_DataSets\SBU_3DFE'};
T=strcat(dataset(1),'.mat');
load(T{1,1});
labelDistribution = labels;
T=strcat(dataset(1),'_binary.mat');
load(T{1,1});

features = zscore(features);

results_all = ones(64,6);
beta_list = [0.0001,0.001,0.01,0.1,1,10,100,1000];
lambda_list = [0.0001,0.001,0.01,0.1,1,10,100,1000];
index = 0;
for i = 1:8
    for j =1:8
        index = index+1;
        fprintf('-----start calculating low rank representation for ground label\n');
        beta = beta_list(i); % tradeoff parameter of LRR
        [C_low_rank,~] = solve_lrr(features',beta);
        fprintf('-----finish calculating low rank representation for ground labels\n');
        
        lambda = lambda_list(j);
        
        [W, numerical] = GLLE(logicalLabel,features,C_low_rank,lambda);
        distribution = (softmax(numerical'))';
        
        results_all(index,1) = chebyshev(distribution,labelDistribution);
        results_all(index,2) = clark(distribution,labelDistribution);
        results_all(index,3) = canberra(distribution,labelDistribution);
        results_all(index,4) = kldist(distribution,labelDistribution);
        results_all(index,5) = cosine(distribution,labelDistribution);
        results_all(index,6) = intersection(distribution,labelDistribution);
        
        fprintf('--------------------------------------\n');
        fprintf('beta:%f, lambda:%f\n',beta,lambda);
        fprintf('\tchebyshev   :%f\n',results_all(index,1));
        fprintf('\tclark       :%f\n',results_all(index,2));
        fprintf('\tcanberra    :%f\n',results_all(index,3));
        fprintf('\tkldist      :%f\n',results_all(index,4));
        fprintf('\tcosine      :%f\n',results_all(index,5));
        fprintf('\tintersection:%f\n',results_all(index,6));
        
    end
end



