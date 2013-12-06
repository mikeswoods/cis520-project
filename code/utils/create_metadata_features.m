load('../data/business_metadata.mat')
load('../data/user_metadata.mat')
load('../data/review_dataset.mat')
load('../data/extra_fields.mat')

N = numel(train.labels);
unique_users = unique(user_id_idx);
unique_businesses = unique(business_id_idx);

average_user_ratings = zeros(N,1);
for i = 1:numel(unique_users)
   user_id = unique_users(i);
   current_user_indexes = find(user_id_idx == user_id);
   average_label_val = round(mean(train.labels(current_user_indexes)));
   
   average_user_ratings(current_user_indexes) = average_label_val;
end

average_biz_ratings = zeros(N,1);
for i = 1:numel(unique_businesses)
   biz_id = unique_businesses(i);
   current_biz_indexes = find(business_id_idx == biz_id);
   average_label_val = round(mean(train.labels(current_biz_indexes)));
   
   average_biz_ratings(current_biz_indexes) = average_label_val;
end

new_features = [ extra_fields average_user_ratings average_biz_ratings ];