function precision1 = precision_at_k(ids, Lbase, Lquery, K,num)


if ~exist('K','var')
	K = size(Lbase,1);
end

nquery = size(ids, 2);
P = zeros(K, nquery);

for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:K), :), label), 2) > 0;
    Lk = cumsum(imatch);
    P(:, i) = Lk ./ (1:K)';
end
precision = mean(P, 2);
n=int16(K/num);
k=1;
for j=1:n:K
    precision1(k)=precision(j);
    k=k+1;
end

end

