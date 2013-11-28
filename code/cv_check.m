function [rmses] = cv_check(X, Y, W, learners, nfolds)
%
% Calculates the root-mean squared error of the given prediction data
% by way of N-fold cross validation.
%
% CV_CHECK(X, Y, W, learners, nfold)
%
% [X] is a N x M matrix of training observations
% [Y] is a N x 1 vector of training labels
% [W] is a K x 1 vector of weights used weigh the prediction results of K
%   learners
% [learners] is a K x 1 vector of function handles, where each handle 
%   represents a learner that accepts 3 arguments and has the following
%   function signature:
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
K = size(learners, 1);

% Divides the training data into N-folds, each as close to equally sized 
% as possible:

cvidx = repmat(1:nfolds, 1, ceil(N / nfolds));
cvidx = cvidx(1:N);
cvidx = cvidx(randperm(N));

rmses = nan(nfolds, 1);
traintimes = nan(nfolds, 1);

for i = 1:nfolds

   tic
   fprintf('%d / %d ', i, nfolds)

   X_train = X(cvidx ~= i, :);
   Y_train = Y(cvidx ~= i);
   
   X_test = X(cvidx == i, :);
   Y_test = Y(cvidx == i);
   N_test_size = size(Y_test, 1);
   
   % Y_predict is an N x K matrix of learner predictions. The k'th column
   % corresponds to the 

   Y_predict = nan(N_test_size, K);
   for j = 1:K
       Y_predict(:, j) = learners{j}(X_train, Y_train, X_test);
   end

   % Find the weighted average of the predicted labels:

   Y_hat = zeros(N_test_size, 1);
   for j = 1:K
       Y_hat = Y_hat + (W(j) .* Y_predict(:, j));
   end
   Y_hat = Y_hat ./ sum(W);

   rmses(i) = sqrt(mean((Y_hat - Y_test) .^ 2));

   fprintf(' RMSE: %.3f\n', rmses(i))

   traintimes(i) = toc;
end

fprintf('Mean RMSE: %.3f\n', mean(rmses))
fprintf('Mean train time: %.3f\n', mean(traintimes))
