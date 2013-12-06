clear;
fprintf('Loading data...\n')
load ../data/review_dataset.mat
%load ../data/small/review_dataset_first_1000.mat
%load ../data/small/review_dataset_first_5000.mat
load ../data/new_features_train.mat

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

clear quiz train

%initialize_additional_features;

%% Get model

addpath packages

models = generate_models(Xt_counts, Yt, 'counts_logit_reg', 'nb', 'counts_svm');
%models = generate_models(horzcat(Xt_counts, sparse(new_features)), Yt, 'counts_logit_reg', 'nb', 'counts_svm');

fprintf('Saving models...\n')

save_models('models.mat', models);

% Xt_additional_features = [];
% Xq_additional_features = [];
initialize_additional_features;

%% Run algorithm

fprintf('Predicting labels...\n')

rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                       Xq_additional_features, Yt);

%% Save results to a text file for submission

dlmwrite('submit.txt', rates,'precision','%d');
