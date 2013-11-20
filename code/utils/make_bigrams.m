function bigrams = make_bigrams(struct,field,vocab)

i = cell(length(struct),1);
j = cell(length(struct),1);
s = cell(length(struct),1);
%nvocab = length(vocab);

tic
for n = 1:100 %length(struct)
   fprintf('%d ',n)
   nwords = length(struct(n).(field));
   
   i{n} = nan(nwords-1,1);
   j{n} = nan(nwords-1,1);
   s{n} = ones(nwords-1,1);
   
   w2 = find(strcmp(struct(n).(field){1},vocab));
   for w = 1:nwords-1
      w1 = w2;
      w2 = find(strcmp(struct(n).(field){w+1},vocab));
      i{n}(w) = w1;
      j{n}(w) = w2;
   end
end
toc

keyboard