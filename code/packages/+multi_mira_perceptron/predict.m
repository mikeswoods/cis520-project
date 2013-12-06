function [Y] = predict(model, X_test, test_idx)
%
% MULTI_MIRA_PERCEPTRON.PREDICT(X_test, model)
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
N = size(X_test, 1);
Y = NaN(N, 1);

for i=1:N
    [~, Y(i)] = max(X_test(i, :) * model.weights);
end

end