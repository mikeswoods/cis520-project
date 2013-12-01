function [rates,Yhat] = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
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

% models.mat should contain a single struct "models". Each field in 
% "models" contains the learner model object used to make predictions
% below. Additionally, models_idx should be defined that specifies the
% index of the given model

load('models/models.mat')

addpath packages

N = size(Xt_counts, 1);
K = numel(models_idx);

Yhat = NaN(N, K);

for i = 1:K

    model_name = models_idx{i};
    fprintf('Predicting for model "%s"...\n', model_name);

    % Get the <model_name>.predict function from eah method as the predictor
    predictor = str2func([model_name '.predict']);

    Yhat(:, i) = predictor(Xt_counts, models.(model_name));
end

% Weight everything equally for now
model_weights = ones(1, K);

%rates = weighted_majority_vote([1 2 3 4 5], model_weights, Yhat);
rates = weighted_average(model_weights, Yhat);
%rates = int8(((3 .* Yhat.nb) + (7 .* Yhat.counts_logit_reg)) ./ 10);

end




