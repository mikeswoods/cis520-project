clear;
fprintf('Loading data...\n')
load ../data/review_dataset.mat
%load ../data/small/review_dataset_first_100.mat

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

clear quiz train

%initialize_additional_features;

%% Get model

addpath packages

% The following models will be run. Each entry is the name of a model 
% package in the code/packages directory.

run_packages = {'counts_logit_reg', 'nb'};
K = numel(run_packages);

for i = 1:K

    pkg_name = run_packages{i};
    fprintf('Learning model "%s"...\n', pkg_name);

    % Get the <package>.train function from eah method as the trainer
    trainer = str2func([pkg_name '.train']);
    
    % Each model has an entry in the model struct given by its package
    % name. It can be accessed dynamically using the sytax
    % <var>.("package-name")
    models.(pkg_name) = trainer(Xt_counts, Yt);
end

fprintf('Saving models...\n')

save_models(models);

Xt_additional_features = [];
Xq_additional_features = [];


%% Run algorithm

fprintf('Predicting labels...')

rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                       Xq_additional_features, Yt);

%% Save results to a text file for submission

dlmwrite('submit.txt', rates,'precision','%d');
