function [model] = train(X_train, Y_train, train_idx, opts)
%
% EIGENWORDS_SVM.TRAIN(train_labels, train_data, opts)
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
if ~exist('opts','var')
   opts = '-q';
end

% svs_file = '../data/eigenwords/svd_10.mat';
% svs = load(svs_file);
% [centroids_U, centroids_V] = eigenwords_svm.make_centroids(X_train, svs.U, svs.V);

svs_file = '../data/eigenwords/N2/eigenwords_top_25.mat';
svs = load(svs_file);
[centroids_U, centroids_V] = eigenwords_svm.make_centroids(X_train, svs.UB, svs.VB);

centroids = [centroids_U centroids_V];

clear svs centroids_U centroids_V;

%addpath libsvm-3-2.17/matlab
svm = svmtrain(Y_train, centroids, [opts ' -t 3']);

model = struct('svm', svm, 'svs_file', svs_file);
end
