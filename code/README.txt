Group: Kernel Sanders
Members: Brendan Callahan, Patrick Crutchley, Michael Woods

To train models, see line 20 of run_submission.m, specifically the call to generate_models as the example.
models = generate_models(Xt_counts, Yt,'funny_cool_useful', 'business_user_classifier2', 'counts_logit_reg','counts_svm', 'nb' );

Then look at the packages folder, each package (folders that start with a +) contain one model. Some are very close to each other, each one is summarized in this readme.

You can run cross validation with: cv_check(Xt_counts, Yt, {'funny_cool_useful', 'business_user_classifier2'}, 10);
Once again, just list the models you want to run x-val against.

You may have to change the paths to relevant files for training runs, we created the whole thing assuming we had access to the data directory one dir up. Also some of data we created for training exceeds the 50mb limit, so obviously that was not included.

*avg_perceptron
INCOMPLETE, ignore

*business_user_classifier/business_classifier/user_classifier
This is a boosting approach to modeling the user/business indexes.

*business_user_classifier2
This is a logistic regression approach using the average ratings for users/businesses.

*counts_logit_reg
This is logistic regression with the Training Counts.

*counts_svm
This is SVM with the Training Counts.

*eigenwords_logit_reg
Bigram matrices were generated from the combined training and quiz sets (using the complete text as found in metadata). We ran SVD on the bigram matrices, generating reduced-dimension left and right singular vectors (U and V respectively) for each word in the vocabulary. Predictors for each document were the average position (centroid) of the document's words in the reduced-dimension space.

This model uses these centroids (in both U- and V-space) as features in a logistic regression classifier.

*eigenwords_svm
See above for eigenword generation. This model uses the centroids as features in a support vector machine classifier.

*funny_cool_useful/funny_cool_useful_user
This is a logistic regression approach using the funny/cool/useful field values, and funny_cool_useful_user combines the average user rating feature into the mix (since users write reviews).

*kmeans
This is kmeans of the top 100 or so principal components reconstructed using SVDS. Predictions done with the minimum distance between the cluster centroids.

*multi_mira_perceptron
This is a multi layer perceptron algorithm using the MIRA classifier for online learning.

*nb
This is Naive Bayes using the Training Counts.
