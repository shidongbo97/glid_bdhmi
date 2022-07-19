function [y, nbrokentris, lcrerr, flaterr, u]= dcflatten_wrap(t, nv, edgelens, opt)

% function [y, nbrokentris, lcrerr]= dcflatten_wrap(t, nv, edgelens, Theta, ufix)
%
% wrapper for dcflatten
%
% Input parameters:
% t - mesh connectivity (list of triplets)
% nv - number of vertices
% edgelens - length of edges in each triangle, given mesh (x,t), can be computed by sqrt( meshFaceEdgeLen2s(x, t) )
% Theta - angle sum for each vertex
% ufix - indices and value of u to be fixed
% for natural boundary condition, Theta and ufix can be ignored

% assert(~exist('mosekopt'), 'Mosek in path, will interferce with fminunc, remove it from path first!');

% [~, y, nbrokentris, flaterr] = evalc( 'dcflatten(t, nv, edgelens, opt)' );
[y, nbrokentris, flaterr] = dcflatten(t, nv, edgelens, opt);

if nargout>2
    %% sanity check
    llcr0 = LCRFromFaceEdgeLens(t, nv, edgelens);
    llcr1 = LCRFromFaceEdgeLens(t, nv, sqrt(meshFaceEdgeLen2s(y, t)));

    lcrerr = norm(llcr0-llcr1, 'inf');
    if lcrerr>1e-4
        warning( 'lcr not reproduced by CETM, err=%f, lscm flaterr=%f', lcrerr, flaterr );
    end
end

function llcr = LCRFromFaceEdgeLens(t, nv, edgelens)

llcr = sparse(t, t(:,[2 3 1]), log(edgelens(:,[2 3 1])./edgelens), nv, nv);
% llcr( ~llcr | ~llcr' ) = 0;
llcr( xor(llcr, llcr') ) = 0;
llcr = llcr + llcr';


function l = meshFaceEdgeLen2s(x, t)

frownorm2 = @(M) sum(M.^2, 2);
l = frownorm2( x(t(:, [2 3 1]), :) - x(t(:, [3 1 2]), :) );
l = reshape( l, [], 3 );