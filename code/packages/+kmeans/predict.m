function [Y] = predict(model, X_test, test_idx)
%
% NB.PREDICT(X_test, model)
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [model] The model object returned from the counts_logit_reg.train() 
%   function
%
% [test_idx] The P x 1 of selected training indices
%
% [Y] A N x 1 vector of predicted labels
%

%count of training items
num_test = size(X_test, 1);

K = numel(model.cluster_predictions);

[U, S, V] = svds(X_test,75);
%variances = diag(S).^2 / (size(X_train,1)-1);
%varExplained = 100 * variances./sum(variances);
%index = 1+sum(~(cumsum(varExplained)>98));
X_test_svd = U(:,1:75)*S(1:75,1:75);

% [W, X_test_svd] = pca_svds(X_test, 75);

Y = zeros(1, num_test);

%get the prediction for each test point
for i = 1:num_test
    test_point = X_test_svd(i,:);
    
    %get the current distances from this point to all centroids
    current_distances = zeros(1,K);
    for j = 1:K
        current_distances(j) = pdist2(test_point, model.centroids(j,:));
    end
    
    %find the closest centroid
    [min_val, min_idx] = min(current_distances);
    
    %use the closest centroids most common value in training
    Y(i) = model.cluster_predictions(min_idx);
end