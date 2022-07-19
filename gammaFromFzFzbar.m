function r = gammaFromFzFzbar(fz, fzbar)

% i = find( abs(fz)>1 );
k = abs(fzbar./fz);
i = find( abs(fz).^2 > max(k)./k );
fz = fz(i);
fzbar = fzbar(i);

k = abs(fzbar./fz);
logn = log( max(k) ) - log(k.*abs(fz).^2);
% gama = logn./wrightOmega( log(logn) - 1 - log(2*abs(log(abs(fz)))));
% gama = real(gama);

m = 2*exp(1)*log(abs(fz));
r = m.*exp( lambertw(-1, logn./m) );

r = max(r);