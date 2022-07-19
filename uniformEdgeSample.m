function i = uniformEdgeSample(x, n)

% try to sample the polygon evenly, while keeping original vertices

assert(~isreal(x));

nx = numel(x);

if n>=nx
    i = 1:nx;
    return
end

edgelens = abs(x - x([2:end 1]));
edgesubs = ceil( cumsum(edgelens)/sum(edgelens)*n );

i = find( edgesubs>edgesubs([end 1:end-1]) );

% fDrawPoly = @(x)drawmesh([1:size(x,1);2:size(x,1),1;1:size(x,1)]',x);
% figuredocked; h = fDrawPoly(x(i));
% set(h, 'marker', 'x');