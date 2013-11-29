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

% models.mat should contain a single struct "models". Each field in 
% "models" contains the learner model object used to make predictions
% below

load('models/models.mat')

addpath packages

run_packages = fieldnames(models);
K = numel(run_packages);

for i = 1:K

    pkg_name = run_packages{i};
    fprintf('Predicting for model "%s"...\n', pkg_name);

    % Get the <package>.predict function from eah method as the predictor
    predictor = str2func([pkg_name '.predict']);
    
    % Each model has an entry in the Yhat struct given by its package
    % name. It can be accessed dynamically using the sytax
    % <var>.("package-name")
    Yhat.(pkg_name) = predictor(Xt_counts, models.(pkg_name));
end

rates = int8(((3 .* Yhat.nb) + (7 .* Yhat.counts_logit_reg)) ./ 10);

end
