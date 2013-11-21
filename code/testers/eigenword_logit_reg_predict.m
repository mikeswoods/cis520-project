function Yhat_elr = eigenword_logit_reg_predict(X_test,model_clr)

N = size(X_test,1);
init = rand(N,1)*5;

addpath liblinear-1.94/matlab

<<<<<<< HEAD
Yhat_elr = predict(rand(N,1)*5,X_test,model_clr);
=======
Yhat_elr = predict(rand(N,1)*5,Xq_counts,model_clr);
>>>>>>> dccc6f03d583084f711d27288904e9ce1c043290
