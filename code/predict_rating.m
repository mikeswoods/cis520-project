function rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                                Xq_additional_features, Yt)
% Returns the predicted ratings, given wordcounts and additional features.
%
% Usage:
%
%   RATES = PREDICT_RATING(XT_COUNTS, XQ_COUNTS, XT_ADDITIONAL_FEATURES, ...
%                         XQ_ADDITIONAL_FEATURES, YT);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of wordcount and additional features and produces a
% ranking matrix as explained in the project overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 10 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

N = size(Xq_counts, 1);

load('models/models.mat')

addpath packages
addpath testers

if exist('model_clr','var')
   Yhat_clr = counts_logit_reg_predict(Xq_counts,model_clr);
end

if exist('model_nb','var')
   Yhat_nb = predict(model_nb,Xq_counts);
end

rates = int8(((3.*Yhat_nb) + (7.* Yhat_clr)) ./ 10);

end
