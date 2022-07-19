function w = linearWeight(n, t)

t2 = t*(n-1)+1;
i = floor(t2);
w = zeros(n, 1);

w(i) = 1+i-t2;
if i<n, w(i+1) = t2-i; end
