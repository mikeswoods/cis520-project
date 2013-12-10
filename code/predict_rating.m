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

N = size(Xq_counts, 1);
K = numel(models_idx);

Yhat = NaN(N, K);

for i = 1:K

    model_name = models_idx{i};
    fprintf('Predicting for model "%s"...\n', model_name);

    % Get the <model_name>.predict function from eah method as the predictor
    predictor = str2func([model_name '.predict']);
    
    if strcmp(model_name, 'funny_cool_useful')
       Yhat(:, i) = predict(ones(N,1), sparse(Xq_additional_features(:,1:3)), models.(model_name));
    elseif strcmp(model_name, 'business_user_classifier2')
       Yhat(:, i) = predict(ones(N,1), sparse(Xq_additional_features(:,4:5)), models.(model_name));
    else%works for xval
       Yhat(:, i) = predictor(models.(model_name), Xq_counts, 1:N); 
    end
    
    %Yhat(:, i) = predictor(models.(model_name), horzcat(Xq_counts, sparse(Xq_additional_features)), 1:N);
end

% Weight everything equally for now
%model_weights = ones(1, K);

%weights as determined via xval avg-regress
model_weights = [-0.065456743911522 0.395292690926297 0.288002138467192 0.0935124821244977 0.238748814098951];
%[0.347245616173471 0.438246295558858 0.163993550999013]

rates = int8(weighted_average(model_weights, Yhat, [1 5]));

end




