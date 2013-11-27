clear;
fprintf('Loading data...\n')
load ../data/review_dataset.mat

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

clear quiz train

%initialize_additional_features;

%% Get model

addpath learners

%X_counts = vertcat(Xt_counts, Xq_counts);

fprintf('Learning models...\n')
% logistic regression on counts
model_clr = counts_logit_reg_train(Yt,Xt_counts);
model_nb = NaiveBayes.fit(Xt_counts, Yt, 'Distribution', 'mn');

fprintf('Saving models...\n')
if exist('models/models.mat','file')
   movefile('models/models.mat',['models/models_backup_' datestr(now,30) '.mat'])
end
save('models/models.mat','-regexp','model_\w*')


Xt_additional_features = [];
Xq_additional_features = [];


%% Run algorithm
fprintf('Predicting labels...')

rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                       Xq_additional_features, Yt);

%% Save results to a text file for submission
dlmwrite('submit.txt',rates,'precision','%d');