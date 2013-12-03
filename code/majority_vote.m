function [vote] = majority_vote(Y_hat)
%
% MAJORITY_VOTE
%
% Simple majority vote implementation
%
[N, K] = size(Y_hat);

if (K == 1)
    vote = mode(Y_hat);
else
  vote = NaN(N, 1);
  for i = 1:N
      vote(i,:) = mode(Y_hat(i,:));
  end
end
end
