function [model] = train(X_train, Y_train, opts)
%
% COUNTS_SVM.TRAIN(train_labels, train_data, opts)
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
%
% [opts] Options used for training the learner. This value is optional
%
% [model] The trained learner model instance
%

default_opts = '-s 1 -q'; % 1 -- L2-regularized L2-loss support vector classification (dual)

if ~exist('opts', 'var')
   opts = default_opts;
else
   opts = [opts default_opts];
end

model = train(Y_train, X_train, opts);
end
