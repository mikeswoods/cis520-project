% mean_rmses = zeros(10,10);
% for i = 1:10
%    for j = 1:10
%       mean_rmses(i,j) = mean(cv_check_blended(train.counts, train.labels, 10, i, j));
%    end
% end

function rmses = cv_check_blended(data,labels,nfolds,pct_nb,pct_lr)

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
   Y_train = labels(cvidx~=i);
   
   X_test = data(cvidx==i,:);
   Y_test = labels(cvidx==i);
   
   clr = counts_logit_reg_train(Y_train,X_train);
   nb = NaiveBayes.fit(X_train,Y_train, 'Distribution', 'mn');
   %keyboard
   
   Yhat_nb = predict(nb,X_test);
   Yhat_clr = counts_logit_reg_predict(X_test,clr);
   Yhat = round(((pct_nb.*Yhat_nb) + (pct_lr.* Yhat_clr)) ./ (pct_nb + pct_lr));
   
   rmses(i) = sqrt(mean((Yhat-Y_test).^2));
   fprintf(' RMSE: %.3f\n',rmses(i))
   traintimes(i) = toc;
end
fprintf('Mean RMSE: %.3f\n',mean(rmses))
fprintf('Mean train time: %.3f\n',mean(traintimes))