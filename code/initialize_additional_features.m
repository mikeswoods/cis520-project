% This is the script to process additional features that you want to use

% YOUR CODE GOES HERE

load ../data/metadata.mat

%get the total number of rows
N = size(Xt_counts,1) + size(Xq_counts,1);

%get the overall average for Yt
overall_label_average = round(mean(Yt));

%concatenate the train/quiz matrices
both_metadata = horzcat(train_metadata, quiz_metadata);

extra_fields = zeros(N,3);
%extract the funny/cool/useful field values
for i = 1:N
   extra_fields(i,:) = [ both_metadata(i).cool both_metadata(i).funny both_metadata(i).useful]; 
end

%concatenate the training/quiz matrices and extract the user/business indexes for training/quiz data
[all_user_idx vals1] = extract_user_ids(both_metadata);
[all_biz_idx vals2] = extract_business_ids(both_metadata);

%get the unique users/businesses
unique_users = unique(all_user_idx);
unique_businesses = unique(all_biz_idx);

%generate all the average user ratings
average_user_ratings = zeros(N,1);
for i = 1:numel(unique_users)
   user_id = unique_users(i);
   current_user_indexes = find(all_user_idx == user_id);
   train_only_user_indexes = current_user_indexes(current_user_indexes <= size(Xt_counts,1));
   average_label_val = round(mean(Yt(train_only_user_indexes)));
   
   if not(isnan(average_label_val))
       average_user_ratings(current_user_indexes) = average_label_val;
   else
       average_user_ratings(current_user_indexes) = overall_label_average;
   end
   
end

%generate all the average biz ratings
average_biz_ratings = zeros(N,1);
for i = 1:numel(unique_businesses)
   biz_id = unique_businesses(i);
   current_biz_indexes = find(all_biz_idx == biz_id);
   train_only_biz_indexes = current_biz_indexes(current_biz_indexes <= size(Xt_counts,1));
   average_label_val = round(mean(Yt(train_only_biz_indexes)));
   
   if not(isnan(average_label_val))
       average_biz_ratings(current_biz_indexes) = average_label_val;
   else
       average_biz_ratings(current_biz_indexes) = overall_label_average;
   end
end

new_features_quiz = [ extra_fields(25001:30000,:) average_user_ratings(25001:30000) average_biz_ratings(25001:30000) ];

Xt_additional_features = []; % Modify this in if needed
Xq_additional_features = new_features_quiz; % Modify this in if needed
