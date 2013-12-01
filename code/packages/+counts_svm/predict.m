function [Y] = predict(X_test, model)
%
% COUNTS_SVM.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the counts_svm.train() 
%   function
% 
% [Y] A N x 1 vector of predicted labels
%
N = size(X_test, 1);
init = rand(N, 1) * 5;

addpath liblinear-1.94/matlab

Y = predict(init, X_test, model);

end