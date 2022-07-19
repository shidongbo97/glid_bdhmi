function [fzfzbar, E] = fzfzbarMesh(x, y, t)

if isreal(x), x = complex(x(:,1), x(:,2)); end
if isreal(y), y = complex(y(:,1), y(:,2)); end

nv = size(x, 1);
nf = size(t, 1);
eInF = x(t(:,[3 1 2])) - x(t(:,[2 3 1]));
A = signedAreas(x, t)*4;

E = sparse( repmat(1:nf, 3, 1)', t, conj(1i*eInF)./[A A A], nf, nv );
fzfzbar = [E*y conj(E)*y];