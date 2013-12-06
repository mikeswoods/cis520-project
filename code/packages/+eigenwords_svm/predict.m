function [Y] = predict(model, X_test, test_idx)
%
% EIGENWORDS_SVM.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the EIGENWORDS_SVM.train() 
%   function
% 
% [Y] A N x 1 vector of predicted labels
%
N = size(X_test, 1);

svs = load(model.svs_file);

[centroids_U, centroids_V] = eigenwords_svm.make_centroids(X_test, svs.UB, svs.VB);
centroids = [centroids_U centroids_V];

clear svs centroids_U centroids_V;

%addpath libsvm-3-2.17/matlab
Y = svmpredict(zeros(N, 1), centroids, model.svm);

end
