function [r, errbnd] = gkq(f, a, b)

% http://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula

x = [0.991455371120813
     0.949107912342759
     0.864864423359769
     0.741531185599394
     0.586087235467691
     0.405845151377397
     0.207784955007898];

xg = [x(2:2:end); -x(2:2:end); 0];
xk = [x; -x; 0];

wg = [ 0.129484966168870
       0.279705391489277
       0.381830050505119
       0.417959183673469 ];

wg = [wg(1:end-1); wg(1:end-1); wg(end)];   

wk = [ 0.022935322010529
       0.063092092629979
       0.104790010322250
       0.140653259715525
       0.169004726639267
       0.190350578064785
       0.204432940075298
       0.209482141084728 ];
wk = [wk(1:end-1); wk(1:end-1); wk(end)];


nsub = 3;
if nsub>1
%     wg = [wg; wg]/2;
%     xg = [xg-1; xg+1]/2;

    wg = repmat(wg, nsub, 1)/nsub;
    wk = repmat(wk, nsub, 1)/nsub;

    offsets = ((0:nsub-1)*2+1)/nsub-1;
    xg = reshape( bsxfun( @plus, xg/nsub, offsets ), [], 1 );
    xk = reshape( bsxfun( @plus, xk/nsub, offsets ), [], 1 );
end

if numel(a)==1
    rg = (b-a)/2*wg'*f( ((b-a)*xg +  b+a)/2 );
    rk = (b-a)/2*wk'*f( ((b-a)*xk +  b+a)/2 );
else
    
    rg = wg(1)*f( ((b-a)*xg(1) +  b+a)/2 );
    rk = wk(1)*f( ((b-a)*xk(1) +  b+a)/2 );
    for i=2:numel(wg)
        rg = rg + wg(i)*f( ((b-a)*xg(i) +  b+a)/2 );
    end
    for i=2:numel(wk)
        rk = rk + wk(i)*f( ((b-a)*xk(i) +  b+a)/2 );
    end

    rg = (b-a)/2.*rg;
    rk = (b-a)/2.*rk;
end

r = rk;
errbnd = (200*abs(rg-rk)).^1.5;
