function [idx, business_ids] = extract_business_ids(metadata)
    N = numel(metadata);
    next_idx = 1;
    idx = zeros(N, 1);
    business_ids = struct();

    for i=1:N
        original_business_id = metadata(i).business_id{1};
        business_id = genvarname(original_business_id);
        if ~isfield(business_ids, business_id)
            business_ids.(business_id) = struct('i', [i], 'idx', next_idx);
            next_idx = next_idx + 1;
        else
            business_ids.(business_id).i = [business_ids.(business_id).i i];
        end
    end
    
    fields = fieldnames(business_ids);

    for k=1:numel(fields);
        entry = business_ids.(fields{k});
        for j=1:numel(entry.i)
            idx(entry.i(j)) = entry.idx;
        end
    end
end

