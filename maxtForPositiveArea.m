function t = maxtForPositiveArea( fz, gz, dfz, dgz )

a = abs(dfz).^2 - abs(dgz).^2;
b = 2*real( conj(fz).*dfz - conj(gz).*dgz );
c = abs(fz).^2 - abs(gz).^2;

% assert( all( c>=-1e-10 ) );

delta = b.^2 - 4*a.*c;

% t = single(inf);
t = a+inf;
i = ~(a>0 & (delta<0 | b>0));
% assert( all(delta(a<0)>=0) );
% i1 = (a>0 & b<0 & delta>=0) | (a<0);
% assert( all(i==i1) );
t(i) = (-b(i)-sqrt(delta(i)))./a(i)/2;

%% take care of special case where a==0 numerically
i2 = abs(a)<1e-20 & b<0;
t(i2) = -c(i2)./b(i2);
t( abs(a)<1e-20 & b>0 ) = inf;

% assert( all(t>=0) );
t = max(min(t), 0);
