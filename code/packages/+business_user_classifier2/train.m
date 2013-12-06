function [model] = train(X_train, Y_train, train_idx, opts)
%
% BUSINESS_USER_CLASSIFIER.TRAIN(train_labels, train_data, opts)
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
%
% [train_idx] The N x 1 of selected training indices
%
% [opts] Options used for training the learner. This value is optional
%
% [model] The trained learner model instance
%
load('../data/new_features_train.mat');

default_opts = '-s 7 -q'; % 7 = 7 -- L2-regularized logistic regression (dual)

if ~exist('opts', 'var')
   opts = default_opts;
else
   opts = [opts default_opts];
end
model = train(Y_train, sparse(new_features(train_idx,4:5)), opts);
end
