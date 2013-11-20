function model = counts_logit_reg_train(train_labels,train_data,opts)

if ~exist('opts','var')
   opts = '-s 7 -q'; % best options through CV
end

addpath liblinear-1.94/matlab

model = train(train_labels,train_data,opts);

% produced RMSE 1.0314 on quiz set