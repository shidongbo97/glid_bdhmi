function L = cotLaplaceFromFaceAngles(angles, t, nv)

% L = - sparse( t(:,[2 3 1]), t(:,[3 1 2]), cot(angles), nv, nv );
% L = L+L';

L = - sparse( t(:,[2 3 1 3 1 2]), t(:,[3 1 2 2 3 1]), cot([angles angles]), nv, nv );

L = spdiags(-sum(L,2), 0, L);