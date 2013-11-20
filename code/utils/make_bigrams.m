function bigrams = make_bigrams(struct,field,vocab)

bigrams = cell(length(struct),1);
nvocab = length(vocab);

for n = 1:length(struct)
   disp(n)
   nwords = length(struct(n).(field));
   bigrams{n} = [repmat(n,nwords-1,1) nan(nwords-1,1) ones(nwords-1,1)];
   w2 = find(strcmp(struct(n).(field){1},vocab));
   for w = 1:nwords-1
      w1 = w2;
      w2 = find(strcmp(struct(n).(field){w+1},vocab));
      bigrams{n}(w,2) = (w1-1)*nvocab+w2;
   end
end

keyboard