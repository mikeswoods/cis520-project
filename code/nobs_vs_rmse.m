function [rmses] = nobs_vs_rmse(X, Y, learner, nreps, obs_pcts, test_pct)
%
% Calculates the root-mean squared error of the given prediction data
% with varying percentage of data used to train model.
%
% nobs_vs_rmse(X, Y, learner, nreps, obs_pcts, test_pct)
%
%   Example invocation:
%
%     nobs_vs_rmse(Xt_counts, Yt, 'nb', 10, .4:.1:.9, .1);
%
% [X] is a N x M matrix of training observations
%
% [Y] is a N x 1 vector of training labels
%
% [learner] is the learner package name or function
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
% [nreps] is an integer > 0 that specifies the number of times each training percentage
%   will be run.
%
% [obs_pcts] is a K x 1 vector of training data percentages for which to get RMSE
%
% [test_pct] is the proportion of data to test on. Note that max(obs_pcts)+test_pct must
%   be less than or equal to 1.

addpath packages;
addpath liblinear-1.94/matlab;

N = size(X, 1);

if ~exist('nreps', 'var')
   nreps = 10;
end
if ~exist('obs_pcts', 'var')
   obs_pcts = .5:.1:.9;
end

if ~exist('test_pct', 'var')
   test_pct = 1-max(obs_pcts);
end
if test_pct > 1-max(obs_pcts);
   error('Test percentage greater than the number of untrained observations.')
end

rmses = nan(length(obs_pcts),nreps);

for i = 1:length(obs_pcts)
   nobs = round(N*obs_pcts(i));
   ntest = round(N*test_pct);
   idx = zeros(N,1);
   idx(1:nobs) = 1;
   idx(nobs+1:nobs+ntest) = 2;
   
   for j = 1:nreps
      idx = idx(randperm(N));
      train_idx = idx==1;
      test_idx = idx==2;
      
      X_train = X(train_idx,:);
      Y_train = Y(train_idx==1);
      
      X_test = X(test_idx==1,:);
      Y_test = Y(test_idx==1);
      
      Y_hat = run_predictions(X_train, Y_train, X_test, train_idx, test_idx, learner);
      
      rmses(i,j) = calc_rsme(Y_hat,Y_test);
   end
end

% % Divides the training data into N-folds, each as close to equally sized
% % as possible:
% 
% cvidx = repmat(1:nfolds, 1, ceil(N / nfolds));
% cvidx = cvidx(1:N);
% 
% if ~exist('nshuffle', 'var')
%    nshuffle = randperm(N);
% end
% 
% cvidx = cvidx(nshuffle);
% constrain_labels_to = [1 5];
% 
% for i = 1:nfolds
%    tic
%    fprintf('%d/%d ', i, nfolds)
%    
%    train_idx = find(cvidx ~= i);
%    X_train = X(cvidx ~= i, :);
%    Y_train = Y(cvidx ~= i);
%    
%    test_idx = find(cvidx == i);
%    X_test = X(cvidx == i, :);
%    Y_test = Y(cvidx == i);
%    
%    Y_hat = run_predictions(X_train, Y_train, X_test, train_idx, test_idx, learner);
%    
%    % -- (1) Weighted average /w constant weights
%    rmses(i, 1) = calc_rsme(weighted_average(W, Y_hat, constrain_labels_to), Y_test);
%    
%    % -- (2) Weighted average with weights determined by linear regression
%    b = regress(Y_test, Y_hat);
%    B_regress(i, :) = b';
%    rmses(i, 2) = calc_rsme(weighted_average(B_regress(i, :), Y_hat, constrain_labels_to), Y_test);
%    
%    % -- (3) Weighted average with weights determined by stepwise regression
%    b = stepwisefit(Y_hat, Y_test, 'display', 'off');
%    B_stepwise(i, :) = b';
%    rmses(i, 3) = calc_rsme(weighted_average(B_stepwise(i, :), Y_hat, constrain_labels_to), Y_test);
%    
%    % --- (4) Majority vote ---
%    rmses(i, 4) = calc_rsme(majority_vote(Y_hat), Y_test);
%    
%    % --- (5) Adaptive weighted vote ---
%    rmses(i, 5) = calc_rsme(weighted_majority_vote(Y_hat, Y), Y_test);
%    
%    fprintf(' RMSE: (avg-const) %.3f, (avg-regress) %.3f, (avg-stepwise) %.3f, (maj) %.3f, (weighted-maj) %.3f\n', ...
%       rmses(i, 1), rmses(i, 2), rmses(i, 3), rmses(i, 4), rmses(i, 5));
%    traintimes(i) = toc;
% end
% 
% fprintf('Mean RMSE avg-const: %.3f\n', mean(rmses(:,1)))
% fprintf('Mean RMSE avg-regress: %.3f\n', mean(rmses(:,2)))
% fprintf('Mean avg-regress weights: %s\n',mat2str(mean(B_regress)))
% fprintf('Mean RMSE avg-stepwise: %.3f\n', mean(rmses(:,3)))
% fprintf('Mean avg-stepwise weights: %s\n',mat2str(mean(B_stepwise)))
% fprintf('Mean RMSE maj: %.3f\n', mean(rmses(:,4)))
% fprintf('Mean RMSE weighted-maj: %.3f\n', mean(rmses(:,5)))
% fprintf('Mean train time: %.3f\n', mean(traintimes))
end

function [val] = calc_rsme(Y_hat, Y_test)
val = sqrt(mean((Y_hat - Y_test) .^ 2));
end

