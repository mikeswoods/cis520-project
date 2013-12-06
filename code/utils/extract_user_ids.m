function [idx, user_ids] = extract_user_ids(metadata)
    N = numel(metadata);
    next_idx = 1;
    idx = zeros(N, 1);
    user_ids = struct();

    for i=1:N
        original_user_id = metadata(i).user_id{1};
        user_id = genvarname(original_user_id);
        if ~isfield(user_ids, user_id)
            user_ids.(user_id) = struct('i', [i], 'idx', next_idx);
            next_idx = next_idx + 1;
        else
            user_ids.(user_id).i = [user_ids.(user_id).i i];
        end
    end
    
    fields = fieldnames(user_ids);

    for k=1:numel(fields);
        entry = user_ids.(fields{k});
        for j=1:numel(entry.i)
            idx(entry.i(j)) = entry.idx;
        end
    end
end

