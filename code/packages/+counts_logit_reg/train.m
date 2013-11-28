function [model] = train(train_labels, train_data, opts)
%
% COUNTS_LOGIT_REG.TRAIN(train_labels, train_data, opts)
%
if ~exist('opts', 'var')
   opts = '-s 7 -q'; % best options through CV
end

addpath liblinear-1.94/matlab

model = train(train_labels, train_data, opts);
