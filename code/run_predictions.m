function  [Y_hat] = run_predictions(X_train, Y_train, X_test, learners)
%
% Runs the test() funtion for each of the given learners, collecting
% the results into the Y_hat matrix
%
% RUN_PREDICTIONS(X_train, Y_train, X_test, learners)
%
% [X_train] is a N x M matrix of training observations
%
% [Y_train] is a N x 1 vector of training labels
%
% [X_test] Q x M matrix of test observation
%
% [learners] is a K x 1 vector of learner package names or function 
%   handles. If an entry is a function handle, the function is expected
%   to accept 3 arguments and have the following signature:
%
%     [Y_predict] = @(X_train, Y_train, X_test) ...  
% 
%   where 
% 
%     [X_train] is an P x M N-fold subset matrix of training data derived
%       from X
%     [Y_train] is a P x 1 N-fold subset matrix of training labels derived 
%       from Y
%     [X_test] is a Q x M matrix of held-out training data derived from X
%     [Y_predict] is a Q x 1 vector of predictions
%
% [Y_hat] is a N x K matrix of predictions, there the k-th column
%  contains predictions for the k-th learner in learners
%
N = size(X_test, 1);
K = numel(learners);

Y_hat = zeros(N, K);

for i = 1:K
    test_func = get_test_function(learners{i});
    Y_hat(:, i) = test_func(X_train, Y_train, X_test);
end

end

function [handle] = get_test_function(name_or_func)
%
% [handle] = GET_TEST_FUNCTION(name_or_func)
%
% [handle] If given a function handle, this function will return the 
%   handle as-is. If given a string specifying a package name, this 
%   function will return a function handle for @<package-name>.test
%
if isa(name_or_func, 'function_handle')
    handle = name_or_func;
else
    handle = str2func([name_or_func '.test']); % Make @<name>.test
end

end
