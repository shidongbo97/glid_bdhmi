function [Q, t12, VV2T] = quadInTriMesh(t, nv)

nf = size(t, 1);

% if any( any( sparse(t, t(:, [2 3 1]), ones(nf, 3), nv, nv)>1 ) )
%     warning('inconsistent triangulation!');
% end

VV2T = sparse(t, t(:, [2 3 1]), repmat(1:nf, 1, 3), nv, nv);
assert( nnz(VV2T)==nf*3, 'inconsistent triangulation!' );

[eij1, eij2] = find(VV2T>0 & VV2T'>0);
eij = unique( sort([eij1 eij2], 2), 'rows' );

t12 = full( VV2T( sub2ind([nv, nv], eij, eij(:, [2 1])) ) );

% matrows2cell = @(m) mat2cell(m, ones(size(m,1),1));
% elm = [cellfun(@setdiff, matrows2cell(t(t12(:,1),:)), matrows2cell(eij)), ...
%        cellfun(@setdiff, matrows2cell(t(t12(:,2),:)), matrows2cell(eij))];
   
elm = [sum(t(t12(:,1),:), 2) sum(t(t12(:,2),:), 2)] - sum(eij,2)*[1 1];

Q = [eij elm];