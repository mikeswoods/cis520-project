function [W] = train(X, Y, opts)
% Trains a multi-class, averaged perceptron model
%
% For a N x D sparse feature matrix X and Nx1 label matrix Y, returns
% averaged D x 1 weight model
%
% [X] N x M matrix of observations
%
% [Y] N x 1 vector of labels
% 
% [opts] Options struct
%
% 'numPasses' is the number of times of passing the whole dataset through perceptron.
% The loop will end after number of passes (numPasses) is reached. If not
% specified, 2 will be used
%
% [W] K x D
% 
if exist('opts', 'var') && isfield(opts, 'numPasses')
    numPasses = opts.numPasses;
else
    numPasses = 2;
end

Ys = unique(Y);
K = numel(Ys);
Y_to_K = zeros(K, 1);

for i=1:K
    Y_to_K(i) = Ys(i);
end

[N,D] = size(X);

% Initialize weights -- each weight vector has D features it
W = 0.0001 * ones(D, K);
%W = ones(D, K) ./ K;

j = 1;

for p=1:numPasses
    
    correct = 0;
    total = 0;
    
    for i=1:N
        x_i = X(i,:);
        y_i = Y(i,:);

        predicted_k = predict_class(W, x_i);
        actual_k = Y_to_K(y_i);

        if mod(total, 100) == 0
            fprintf('pass = %d/%d | correct = %f\n', p, i, correct / total);
        end
        
        if predicted_k == actual_k
            % If correct, no change
            correct = correct + 1;
        else
            % Incorrect -- adjust weights of actual_k up, the rest down
            W(:,actual_k) =  W(:,actual_k) + ((predicted_k - actual_k) .^2 * x_i');
            for z=1:K
                if z ~= actual_k
                    W(:,z) = W(:,z) - ((predicted_k - actual_k) ./ 2 * x_i');
                end
            end
        end

        total = total + 1;
        j = j + 1;
    end
end
end

function [final_w] = predict_class(W, x_i)
%
% From http://www.seas.upenn.edu/~cis520/lectures/perceptrons.pdf
%
    K = size(W, 2);
    results = zeros(1, K);

    % Want to select W_k s.t. Yhat = argmax(forall k in Y, W_k * f(x_i))
    for k=1:K
        results(:,k) = x_i * W(:,k);
    end

    [~, final_w] = max(results);
end

