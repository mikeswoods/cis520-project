function [Y_predict] = test(X_train, Y_train, X_test)
%
% This is the function that is passed to the cross-validation testing
% code
%
% EIGENWORDS_SVM.TEST(X_train, Y_train, X_test)
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
%
% [X_test] A N x M matrix of test data, where N is the number of
%   observations, and M is the number of features
%
% [Y_predict] A N x 1 vector of predicted labels
%

model = eigenwords_svm.train(X_train, Y_train);

Y_predict = eigenwords_svm.predict(model, X_test);

end