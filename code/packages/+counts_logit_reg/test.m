function [Y_predict] = test(X_train, Y_train, X_test, train_idx, test_idx)
%
% This is the function that is passed to the cross-validation testing
% code
%
% COUNTS_LOGIT_REG.TEST(X_train, Y_train, X_test)
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [train_idx] The Q x 1 of selected training indices
%
% [test_idx] The P x 1 of selected training indices
%
% [Y_predict] A N x 1 vector of predicted labels
%

model = counts_logit_reg.train(X_train, Y_train, train_idx);

Y_predict = counts_logit_reg.predict(model, X_test, test_idx);

end
