function [Y] = predict(X_test, model)
%
% COUNTS_LOGIT_REG.TRAIN(X_test, model)
%
N = size(X_test, 1);
init = rand(N, 1) * 5;

addpath liblinear-1.94/matlab

Y = predict(init, X_test, model);
