function Yhat_clr = counts_logit_reg_predict(X_test,model_clr)

N = size(X_test,1);
init = rand(N,1)*5;

addpath liblinear-1.94/matlab

Yhat_clr = predict(init,X_test,model_clr);
