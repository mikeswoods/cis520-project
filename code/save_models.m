function save_models(filename, models)
%
% SAVE_MODELS(models)
% 
% Saves The specified variable names to models/models.mat, generating
% a model.mat backup file in the process
% 
% [filename] The filename to save the models to
% 
% [models] A struct, where each field corresponds to each learner
%   model to be saved
%

if exist('models/models.mat', 'file')
   movefile('models/models.mat', ['models/models_backup_' datestr(now,30) '.mat'])
end

save(['models/' filename], 'models');

end
