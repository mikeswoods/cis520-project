% kmeans_rmses = zeros(50,1);
% [U, S, V] = svds(train.counts,100);
% variances = diag(S).^2 / (size(train.counts,1)-1);
% varExplained = 100 * variances./sum(variances);
% index = 1+sum(~(cumsum(varExplained)>98));
% newX = U(:,1:100)*S(1:100,1:100);
% 
% for i = 10:50
%     kmeans_rmses(i) = mean(cv_check_kmeans(newX, train.labels, 10, i));
% end

function rmses = cv_check_kmeans(data,labels,nfolds,K)

if ~exist('nfolds','var')
   nfolds = 5;
end
nobs = size(data,1);

cvidx = repmat(1:nfolds,1,ceil(nobs/nfolds));
cvidx = cvidx(1:nobs);
cvidx = cvidx(randperm(nobs));

rmses = nan(nfolds,1);
traintimes = nan(nfolds,1);

Yhat_init = labels(randperm(nobs));

for i = 1:nfolds
   tic
   fprintf('%d/%d ',i,nfolds)
   X_train = data(cvidx~=i,:);
   
   Y_test = labels(cvidx==i);
   
   N = numel(Y_test);
   
   [clusters,C] = kmeans(X_train,K);
   
   cluster_predictions = zeros(1,K);
   for j = 1:K
       indices = find(clusters == j);
       zero_count = numel(indices);
       one_count = numel(indices);
       if zero_count > one_count
           cluster_predictions(j) = 0;
       else
           cluster_predictions(j) = 1;
       end
   end
   
   Yhat = zeros(N, 1);
   for j = 1:N
       Yhat(j) = cluster_predictions(clusters(j));
   end
   
   rmses(i) = sqrt(mean((Yhat-Y_test).^2));
   fprintf(' RMSE: %.3f\n',rmses(i))
   traintimes(i) = toc;
end
fprintf('Mean RMSE: %.3f\n',mean(rmses))
fprintf('Mean train time: %.3f\n',mean(traintimes))