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

Y = model.model.predict(model.metadata(test_idx,:));
end