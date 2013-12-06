function [model] = train(X_train, Y_train, train_idx, opts)
%
% USER_CLASSIFIER.TRAIN(train_labels, train_data, opts)
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
%
% [train_idx] The N x 1 of selected training indices
%
% [opts] Options used for training the learner. This value is optional
%
% [model] The trained learner model instance
%
metadata = load('../data/user_metadata.mat');

mdl = fitensemble(metadata.user_id_idx(train_idx,:), Y_train, 'LSBoost', 5, 'tree');

model = struct('metadata', metadata, 'model', mdl);
end
