function sanitized = sanCellArray(input)

if ~isvector(input)
   error('input is not a vector array')
end

sanitized = regexprep(lower(input),'[^a-z0-9]','');
