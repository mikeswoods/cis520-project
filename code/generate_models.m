function [models] = generate_models(X_train, Y_train, varargin)
%
% [models] = GENERATE_MODELS(...) 
%
% GENERATE_MODELS accepts a variable number of models names and
% corresponding options.
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
% 
% [...] Each argument following [Y_train] can be either a string
%   specifying an algorithm package listed in code/packages to train, or a
%   2 element cell of the form {'model-package-name', <opts>}
%
% [models] Is a struct, where each entry is the name of a model, and the
%   value is the model object itself
%
addpath packages

K = numel(varargin);
N = numel(Y_train);

for i = 1:K
    
    k_args = varargin{i};
    
    % If a cell, it should be of the form {'model-package-name', <opts>}

    if iscell(k_args)
        pkg_name = k_args{1};
        k_run_with = {k_args{2}};
    else
        pkg_name = k_args;
        k_run_with = {};
    end

    fprintf('Learning model "%s"...\n', pkg_name);

    % Get the <package>.train function from eah method as the trainer
    trainer = str2func([pkg_name '.train']);
    
    % Each model has an entry in the model struct given by its package
    % name. It can be accessed dynamically using the sytax
    % <var>.("package-name")
    models.(pkg_name) = trainer(X_train, Y_train, 1:N, k_run_with{:});
end

end

