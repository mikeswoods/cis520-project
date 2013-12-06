function [Y] = predict(model, X_test, test_idx)
%
% EIGENWORDS_LOGIT_REG.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the EIGENWORDS_SVM.train() 
%   function
%
% [test_idx] The P x 1 of selected training indices
%
% [Y] A N x 1 vector of predicted labels
%

svs = load(model.svs_file);
[centroids_U, centroids_V] = eigenwords_logit_reg.make_centroids(X_test, svs.UB, svs.VB);
centroids = [centroids_U centroids_V];
clear svs centroids_U centroids_V;

R = mnrval(model.B, centroids);
[~, Y] = max(R, [], 2);
end
