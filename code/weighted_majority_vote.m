function [Y_final] = weighted_majority_vote(Y_hat, Y, W_init)
%
% [W] = ADAPTIVE_WEIGHT
%
% [Y_hat]
%
% [Y]
%
% [W_init]
%
% [Y_final]
%
[N, K] = size(Y_hat);

if ~exist('W_init', 'var')
    W_init = ones(1, K) .* (1 / K);
end

W = W_init;

for i=1:N
    cmp = Y_hat(i,:) == Y(i,:);
    correct = find(cmp == 1);
    incorrect = find(cmp == 0);

    step = 1 ./ mean(W);

    W(correct) = W(correct) + step;
    W(incorrect) = W(incorrect) - step;
end

Y_final = round(weighted_average(W, Y_hat, [1 5]));

end

