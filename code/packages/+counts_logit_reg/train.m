function [model] = train(X_train, Y_train, opts)
%
% COUNTS_LOGIT_REG.TRAIN(train_labels, train_data, opts)
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
if ~exist('opts', 'var')
   opts = '-s 7 -q'; % best options through CV
end

addpath liblinear-1.94/matlab

model = train(Y_train, X_train, opts);
end