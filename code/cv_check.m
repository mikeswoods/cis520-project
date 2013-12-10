function [rmses] = cv_check(X, Y, learners, nfolds, W, nshuffle)
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
%
% [W] is a K x 1 vector of weights used weigh the prediction results of K
%   learners. This is noly used when combine = 'average'. If omitted,
%   equal weights will be used

addpath packages;
addpath liblinear-1.94/matlab;

N = size(X, 1);
K = numel(learners);

rmses = nan(nfolds, 4);
B_regress = nan(nfolds, K); % Linear Regression weights
B_stepwise = nan(nfolds, K); % Stepwise Regression weights
traintimes = nan(nfolds, 1);

if ~exist('nfolds', 'var')
   nfolds = 5;
end

if ~exist('W', 'var')
    W = ones(1, K);
end

% Divides the training data into N-folds, each as close to equally sized 
% as possible:

cvidx = repmat(1:nfolds, 1, ceil(N / nfolds));
cvidx = cvidx(1:N);

if ~exist('nshuffle', 'var')
    nshuffle = randperm(N);
end

cvidx = cvidx(nshuffle);
constrain_labels_to = [1 5];

for i = 1:nfolds
   tic
   fprintf('%d/%d ', i, nfolds)

   train_idx = find(cvidx ~= i);
   X_train = X(cvidx ~= i, :);
   Y_train = Y(cvidx ~= i);

   test_idx = find(cvidx == i);
   X_test = X(cvidx == i, :);
   Y_test = Y(cvidx == i);

   Y_hat = run_predictions(X_train, Y_train, X_test, train_idx, test_idx, learners);

   for j=1:numel(learners)
      fprintf('[%s] Correct = %f\n', learners{j}, mean(Y_hat(:,j) == Y_test));
   end

   % -- (1) Weighted average /w constant weights
   rmses(i, 1) = calc_rsme(weighted_average(W, Y_hat, constrain_labels_to), Y_test);

   % -- (2) Weighted average with weights determined by linear regression
   b = regress(Y_test, Y_hat);
   B_regress(i, :) = b';
   rmses(i, 2) = calc_rsme(weighted_average(B_regress(i, :), Y_hat, constrain_labels_to), Y_test);

   % -- (3) Weighted average with weights determined by stepwise regression
   b = stepwisefit(Y_hat, Y_test, 'display', 'off');
   B_stepwise(i, :) = b';
   rmses(i, 3) = calc_rsme(weighted_average(B_stepwise(i, :), Y_hat, constrain_labels_to), Y_test);
   
   % --- (4) Majority vote ---
   rmses(i, 4) = calc_rsme(majority_vote(Y_hat), Y_test);

   % --- (5) Adaptive weighted vote ---
   rmses(i, 5) = calc_rsme(weighted_majority_vote(Y_hat, Y), Y_test);
   
   fprintf(' RMSE: (avg-const) %.3f, (avg-regress) %.3f, (avg-stepwise) %.3f, (maj) %.3f, (weighted-maj) %.3f\n', ...
       rmses(i, 1), rmses(i, 2), rmses(i, 3), rmses(i, 4), rmses(i, 5));
   traintimes(i) = toc;
end

fprintf('Mean RMSE avg-const: %.3f\n', mean(rmses(:,1)))
fprintf('Mean RMSE avg-regress: %.3f\n', mean(rmses(:,2)))
fprintf('Mean avg-regress weights: %s\n',mat2str(mean(B_regress)))
fprintf('Mean RMSE avg-stepwise: %.3f\n', mean(rmses(:,3)))
fprintf('Mean avg-stepwise weights: %s\n',mat2str(mean(B_stepwise)))
fprintf('Mean RMSE maj: %.3f\n', mean(rmses(:,4)))
fprintf('Mean RMSE weighted-maj: %.3f\n', mean(rmses(:,5)))
fprintf('Mean train time: %.3f\n', mean(traintimes))
end

function [val] = calc_rsme(Y_hat, Y_test)
    val = sqrt(mean((Y_hat - Y_test) .^ 2));
end

