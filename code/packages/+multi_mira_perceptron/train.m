function [model] = train(X_train, Y_train, train_idx, opts)
%
% Trains a multi-class, MIRA perceptron model
%
% MULTI_MIRA_PERCEPTRON.TRAIN(X_train, Y_train, train_idx, opts)
%
% [X_train] A N x M matrix of training data, where N is the number of
%   observations, and M is the number of features
%
% [Y_train] A N x 1 vector of training labels
%
% [train_idx] The N x 1 of selected training indices
%
% [opts] Options used for training the learner. This value is optional
%
% [model] The trained learner model instance
%
% 'numPasses' is the number of times of passing the whole dataset through perceptron.
% The loop will end after number of passes (numPasses) is reached. If not
% specified, 2 will be useds
% 
if exist('opts', 'var') && isfield(opts, 'numPasses')
    numPasses = opts.numPasses;
else
    numPasses = 2;
end

Ys = unique(Y_train);
K = numel(Ys);
Y_to_K = zeros(K, 1);

for i=1:K
    Y_to_K(i) = Ys(i);
end

[N,D] = size(X_train);

% Initialize weights -- each weight vector has D features it
W = 0.001 * ones(D, K);
%W = ones(D, K) ./ K;

j = 1;

for p=1:numPasses
    
    correct = 0;
    total = 0;
    
    fprintf('MULTI_MIRA_PERCEPTRON: Pass %d of %d\n', p, numPasses);

    for i=1:N
        x_i = X_train(i,:);
        x_i_T = x_i';
        y_i = Y_train(i,:);

        [~, predicted_k] = max(x_i * W);

        actual_k = Y_to_K(y_i);

%         if mod(total, 1000) == 0
%             fprintf('pass = %d of %d, %d | correct = %f\n', p, numPasses, i, correct / total);
%         end

        if predicted_k == actual_k
            % If correct, no change
            correct = correct + 1;
        else
            % Incorrect -- adjust weights of actual_k up, the rest down
            W_actual = W(:,actual_k);
            W(:,actual_k) =  W_actual + (tau(x_i, W_actual, W_actual, 0.001) .* x_i_T);
            
            for k=1:K
                if k ~= actual_k
                    W_k = W(:,k);
                    W(:,k) = W_k - (tau(x_i, W_k, W_actual, 0.5) .* x_i_T);
                end
            end
        end

        total = total + 1;
        j = j + 1;
    end
end

model = struct('weights', W, 'Y_to_K', Y_to_K, 'K', K);
end

function [tau_value] = tau(x_i, W_wrong, W_correct, C)
    tau_value = min(C, ((x_i * (W_wrong - W_correct)) + 1) ./ (2 * norm(x_i) .^ 2));
end
