function [model] = train(X_train, Y_train, opts)
    %
    % EIGENWORDS_KMEANS.TRAIN(train_labels, train_data, opts)
    %
    % [X_train] A N x M matrix of training data, where N is the number of
    %   observations, and M is the number of features
    %
    % [Y_train] A N x 1 vector of training labels
    %
    % [opts] Options used for training the learner. This value is optional
    %
    % [model] The trained learner model instance, a cell of the form
    %   {predictions (N x 1), centroid locations (M x K, where M is the
    %    number of clusteres formed)}
    %
    [N, K] = size(X_train);
    M = numel(unique(Y_train)); % # of unique labels
    km_opts = statset('MaxIter', 500);
    
    if ~exist('opts', 'var')
        opts = {'Distance', 'cityblock', 'Start', 'sample'};
    end
    
    [IDX, C] = kmeans(X_train, M, 'Options', km_opts, opts{:});
    
    % -- Generate the counts for IDX(i) <-> label (Y(i):
    LC = zeros(M, M);
    
    size(IDX)
    size(Y_train)
    
    for j=1:N
        LC(IDX(j), Y_train(j)) = LC(IDX(j), Y_train(j)) + 1;
    end
    
    % -- Pick the most frequent letter as the label:
    freq = zeros(M, 1);
    for j=1:M
        [~, label] = max(LC(j,:));
        freq(j) = label;
    end
    
    model = {freq(IDX), C};
end
