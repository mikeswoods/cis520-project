function [centroids_U, centroids_V] = make_centroids(X, U, V)
% Calculates the "U", and "V" centroids in eigenspace for the given U and V
% eigenword matrices
% 
% [X] N x K data
%
% [U] N x K, "U" eigenword matrix
%
% [V] N x K, "V" eigenword matrix  
%
% [centroids_U, centroids_V] Centroids for "U" and "V", respectively
%
N = size(X, 1);
K = size(U, 2);

centroids_U = NaN(N, K);
centroids_V = NaN(N, K);

for i = 1:N
   R = repmat(X(i,:), K, 1)';
   W_i = sum(X(i,:));
   centroids_U(i,:) = sum(R .* U) / W_i;
   centroids_V(i,:) = sum(R .* V) / W_i;
end
end

function [scaled] = scale(data)
%
% Code taken from the LIBSVM FAQ:
% http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f802
%
% Scales the centroids to [0,1]
%
scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
end