Group: Kernel Sanders
Members: Brendan Callahan, Patrick Crutchley, Michael Woods

To train models, see line 20 of run_submission.m, specifically the call to generate_models as the example.
models = generate_models(Xt_counts, Yt,'funny_cool_useful', 'business_user_classifier2', 'counts_logit_reg','counts_svm', 'nb' );

Then look at the packages folder, each package (folders that start with a +) contain one model. Some are very close to each other, each one is summarized in this readme.

You can run cross validation with: cv_check(Xt_counts, Yt, {'funny_cool_useful', 'business_user_classifier2'}, 10);
Once again, just list the models you want to run x-val against.

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

*eigenwords_svm

*funny_cool_useful/funny_cool_useful_user
This is a logistic regression approach using the funny/cool/useful field values, and funny_cool_useful_user combines the average user rating feature into the mix (since users write reviews).

*kmeans
This is kmeans of the top 100 or so principal components reconstructed using SVDS. Predictions done with the minimum distance between the cluster centroids.

*multi_mira_perceptron

*nb
This is Naive Bayes using the Training Counts.
