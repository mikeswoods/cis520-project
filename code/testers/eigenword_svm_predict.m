function Yhat_esvm = eigenword_svm_predict(X_test,model_esvm)

n = size(X_test,1);
init = round(rand(n,1)*4)+1;

svs = load('../data/svd_10.mat');

k = size(svs.U,2);

centroids_U = nan(n,k);
centroids_V = nan(n,k);

for i = 1:n
%    if mod(i,1000)==0
%       fprintf('%d ',i)
%    end
   w = sum(X_test(i,:));
   centroids_U(i,:) = sum(repmat(X_test(i,:),k,1)'.*svs.U)/w;
   centroids_V(i,:) = sum(repmat(X_test(i,:),k,1)'.*svs.V)/w;
end

addpath libsvm-3-2.17/matlab

Yhat_esvm = svmpredict(init,[centroids_U centroids_V],model_esvm);