function [rmses] = cv_check(X, Y, W, learners, nfolds, nshuffle)
%
% Calculates the root-mean squared error of the given prediction data
% by way of N-fold cross validation.
%
% CV_CHECK(X, Y, W, learners, nfold)
%
%   Example invocation:
%
%     cv_check(Xt_counts, Yt, [0.7 0.3], {'nb' 'counts_logit_reg'}, 10);
%
% [X] is a N x M matrix of training observations
%
% [Y] is a N x 1 vector of training labels
%
% [W] is a K x 1 vector of weights used weigh the prediction results of K
%   learners
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
% [nfolds] is an integer > 0 that specifies the number of 
%   cross-validationfolds that X and Y should be divided into. If an
%   explicit value for nfold is omitted, a default value of 5 will be used.

    if ~exist('nfolds', 'var')
       nfolds = 5;
    end

    N = size(X, 1);
    K = numel(learners);

    % Divides the training data into N-folds, each as close to equally sized 
    % as possible:

    cvidx = repmat(1:nfolds, 1, ceil(N / nfolds));
    cvidx = cvidx(1:N);

    if ~exist('nshuffle', 'var')
        nshuffle = randperm(N);
    end

    cvidx = cvidx(nshuffle);

    rmses = nan(nfolds, 1);
    traintimes = nan(nfolds, 1);

    for i = 1:nfolds

       tic
       fprintf('%d/%d ', i, nfolds)

       X_train = X(cvidx ~= i, :);
       Y_train = Y(cvidx ~= i);

       X_test = X(cvidx == i, :);
       Y_test = Y(cvidx == i);
       N_test_size = size(Y_test, 1);

       % Weighted average:

       Y_hat = zeros(N_test_size, 1);
       for j = 1:K
           test_func = get_test_function(learners{j});
           Y_hat = Y_hat + (W(j) .* test_func(X_train, Y_train, X_test));
       end
       Y_hat = round(Y_hat ./ sum(W));

       rmses(i) = sqrt(mean((Y_hat - Y_test) .^ 2));

       fprintf(' RMSE: %.3f\n', rmses(i))

       traintimes(i) = toc;
    end

    fprintf('Mean RMSE: %.3f\n', mean(rmses))
    fprintf('Mean train time: %.3f\n', mean(traintimes))
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
