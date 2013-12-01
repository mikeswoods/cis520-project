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

file_path = ['models/' filename];

if exist(file_path, 'file')
   movefile(file_path, ['models/models_backup_' datestr(now,30) '.mat'])
end

% A numeric index assigned to each model as well:
models_idx = fieldnames(models);

save(file_path, 'models', 'models_idx');

end
