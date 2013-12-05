function [Y] = predict(model, X_test)
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
init = round(rand(N, 1) * 4) + 1;

% model should contain the following 2 fields:
% - 'svs_file' : The path from this package to the matrices formed from svds() that was loaded
%    by EIGENWORDS_SVM.TRAIN
% - 'svm' : The actual svm model returned from svmtrain()

svs = load(model.svs_file);
[centroids_U, centroids_V] = eigenwords_svm.make_centroids(X_test, svs.U, svs.V);
clear svs;

addpath libsvm-3-2.17/matlab
Y = svmpredict(init, [centroids_U centroids_V], model.svm);

% addpath liblinear-1.94/matlab
% Y = predict(init, X_test, model);

end
