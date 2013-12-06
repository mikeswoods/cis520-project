function [Y] = predict(model, X_test, test_idx)
%
% COUNTS_LOGIT_REG.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the counts_logit_reg.train() 
%   function
%
% [test_idx] The P x 1 of selected training indices
%
% [Y] A N x 1 vector of predicted labels
%
N = size(X_test, 1);
init = rand(N, 1) * 5;

Y = predict(init, X_test, model);

end