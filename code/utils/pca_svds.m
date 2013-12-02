function [W, PC, err] = pca_svds(X, k)
%
% Computes the PCA, returning top k principal components for the given 
% matrix X using svds().
%
% [W, PC] = PCA_SVDS(X, k)
%
% X can be reconstructed like so:
%
%   X_r = (PC * W') + (repmat(mean(X), size(X, 1), 1));
%
% [X] N x M input matrix
% 
% [k] The maximum number of PCs to return. If omitted M is used
%
% [W] A K x K matrix The top k coefficients/weights
%
% [PC] A N x K matrix of the top k principal components
%
% [err] The reconstruction error
%
if ~exist('k', 'var')
    k = size(X, 2); % k = M
end
X_bar = repmat(mean(X), size(X, 1), 1);
X_c = X - X_bar;
[~, ~, W] = svds(X_c, k);

PC = X_c * W;

% Finally, calculate the reconstruction error:
X_hat = (PC * W') + X_bar; 
err = norm(X - X_hat, 'fro') .^ 2 ./ norm(X - X_bar, 'fro') .^ 2;
end
