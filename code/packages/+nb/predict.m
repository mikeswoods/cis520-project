function [Y] = predict(X_test, model)
%
% NB.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the counts_logit_reg.train() 
%   function
% 
% [Y] A N x 1 vector of predicted labels
%

Y = predict(model, X_test);

end