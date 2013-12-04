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

addpath(genpath(strcat(pwd, '\liblinear-1.94')))
addpath(genpath(strcat(pwd, '\libsvm-3-2.17')))

addpath(genpath(strcat(pwd, '\packages')))

K = numel(model.models_idx);
N = 1;

Yhat = NaN(N, K);

for i = 1:K
    mypath = path;
    model_name = model.models_idx{i};
    %save(strcat('/home1/b/bcal/tmp/',model_name,'.mat'), 'model_name', 'i','mypath')

    % Get the <model_name>.predict function from eah method as the predictor
    predictor = str2func([model_name '.predict']);
    
    Yhat(i) = predictor(model.models.(model_name), test_words);
end

% Weight everything equally for now
model_weights = ones(1, K);

prediction = int8(weighted_average(model_weights, Yhat));