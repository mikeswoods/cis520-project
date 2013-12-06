function [model] = train(X_train, Y_train, train_idx, opts)
%
% NB.TRAIN(train_labels, train_data, opts)
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
model = NaiveBayes.fit(X_train, Y_train, 'Distribution', 'mn');

end
