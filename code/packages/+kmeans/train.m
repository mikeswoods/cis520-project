function [model] = train(X_train, Y_train, opts)
%
% NB.TRAIN(train_labels, train_data, opts)
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
default_K = 50; % K = 50

if ~exist('opts', 'var')
   K = default_K;
else
   K = opts;
end

[U, S, V] = svds(X_train,75);
%variances = diag(S).^2 / (size(X_train,1)-1);
%varExplained = 100 * variances./sum(variances);
%index = 1+sum(~(cumsum(varExplained)>98));
X_train_svd = U(:,1:75)*S(1:75,1:75);

% [W, X_train_svd] = pca_svds(X_train, 75);

[clusters,centroids] = kmeans(X_train_svd,K, 'Options', statset('MaxIter', 500));

%get the predicted value for each cluster
cluster_predictions = zeros(1,K);
for j = 1:K
    current_cluster_indexes = find(clusters == j);
    current_cluster_y_values = Y_train(current_cluster_indexes);
    
    [value_counts,values] = hist(current_cluster_y_values, unique(current_cluster_y_values));

    [max_val, max_idx] = max(value_counts);
    
    cluster_predictions(j) = values(max_idx);
end

model = struct('centroids', centroids, 'cluster_predictions', cluster_predictions);

end
