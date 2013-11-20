function stemmed = stemCellArray(input)

if ~isvector(input)
   error('input is not a vector array')
end

stemmed = input;

for i = 1:length(input)
   try
      stemmed{i} = porterStemmer(stemmed{i});
   end
end