function model = init_model(vocab)

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
% 

addpath(strcat(genpath(fullfile(pwd, 'liblinear-1.94')), genpath(fullfile(pwd, 'libsvm-3-2.17')), genpath(fullfile(pwd, 'packages'))))

model = load('models/models-.9446-nokmeans.mat');
model.nb_mat = cell2mat(model.models.nb.Params);
