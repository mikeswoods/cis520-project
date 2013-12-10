function [Y] = predict(model, X_test, test_idx)
%
% BUSINESS_USER_CLASSIFIER.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the <package>.train() 
%   function
%
% [test_idx] The P x 1 of selected training indices
%
% [Y] A N x 1 vector of predicted labels
%

load('../data/new_features_train.mat');

N = size(X_test, 1);
init = rand(N, 1) * 5;

Y = predict(init, sparse(new_features(test_idx,4:5)), model);
end