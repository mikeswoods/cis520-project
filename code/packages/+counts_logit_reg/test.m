function [Y_predict] = test(X_train, Y_train, X_test)
% This is the function that is passed to the cross-validation testing
% code
%
% COUNTS_LOGIT_REG.TRAIN(X_test, model)
%
% [X_train] is an P x M N-fold subset matrix of training data derived
%   from X
% [Y_train] is a P x 1 N-fold subset matrix of training labels derived 
%   from Y
% [X_test] is a Q x M matrix of held-out training data derived from X
% [Y_predict] is a Q x 1 vector of predictions

clr = counts_logit_reg.train(Y_train, X_train);
Y_predict = counts_logit_reg.predict(X_test, clr);

end
