function prediction = make_final_prediction(model,test_words,test_meta)

% Input
% test_words : a 1xp vector representing "1" test sample.
% test_meta : a struct containing the metadata of the test sample.
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

K = numel(model.models_idx);
N = 1;

Yhat = NaN(N, K);

for i = 1:K
    model_name = model.models_idx{i};

    % Get the <model_name>.predict function from each method as the predictor
    predictor = str2func([model_name '.predict']);
    if strcmp(model_name,'nb')
        %send as a third parameter our precalculated matrix, to speed things up
        Yhat(i) = predict_nb(model.models.(model_name), test_words, model.nb_mat);
    else
        % Passed index is not known, so use NaN
        Yhat(i) = predictor(model.models.(model_name), test_words, NaN);
    end
    
end

% Weight everything equally for now
model_weights = ones(1, K);

% Model weights should be substituted with experimentally determined weights
% via cv_check. NOTE: It is important to keep the order of the weights the same as the
% as the learner it is associated with 

% b = regress(Y_test, Y_hat);
% B_regress(i, :) = b';

prediction = int8(weighted_average(model_weights, Yhat, [1 5]));