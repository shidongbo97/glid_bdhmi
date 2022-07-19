function [Z, sqrelens] = metricInterp(x, t, w, opt)

if nargin<4
    opt = struct( 'metric', 'metric tensor', 'hf', -1 );
end

if isfield(opt, 'anchors') && ~isempty(opt.anchors)
    anchId = opt.anchors;
else
    anchId = diameter(x{1});
end

assert( numel(anchId)~=1 );

doTensor = strcmp(opt.metric, 'metric tensor');
showmorph = isfield(opt, 'hf') && ishandle(opt.hf);

if showmorph
    if isfield(opt, 'fMC')
        fMeshColoring = opt.fMC;
    else
        fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'flat', 'EdgeAlpha', 0.02, 'FaceVertexCData', cdata);
    end
    fPlotMarkers = @(x) plot( x(:,1), x(:,2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');

    if ~isfield( opt, 'pause' ); opt.pause = 0; end
    pt = opt.pause;
    
    cla; 
    h(1) = drawmesh(t, x{1}); title( 'BD Morph' ); hold on; 
    fMeshColoring(h(1), x{1}(:,2)); axis on;
    h(2) = fPlotMarkers( x{1}(anchId,:) );
end

%%
nx = numel(x);
nv = size(x{1}, 1);
nw = size(w,1);
Z = cell(nw,1);
sqrelens = cell(nw,1);

xelen2s = cellfun(@(x) meshFaceEdgeLen2s(x, t), x, 'UniformOutput', false);

% for edge length interpolation
if ~doTensor
    xelen2s = cellfun(@sqrt, xelen2s, 'UniformOutput', false);
end

fc2r = @(x) [real(x) imag(x)];
fr2c = @(x) complex(x(:,1), x(:,2));
fNormalize = @(x) x./abs(x);

%%
for iw=1:nw
    %% blend metric tensor or edge lengths acoording to doTensor
    % todo optimize use matrix vector operations
    sqrelens{iw} = xelen2s{1}*w(iw, 1);
    for j=2:nx
        sqrelens{iw} = sqrelens{iw}+xelen2s{j}*w(iw, j);
    end

    if doTensor
        elens = sqrt( sqrelens{iw} );
    else
        elens = sqrelens{iw};
        sqrelens{iw} = elens.^2;
    end

    %% CETM embedding
    [z, nbrokentris, dcferr, flaterr] = dcflatten_wrap(t, nv, elens, struct('numMaxIters', 50));   % natural boundary condition
%     z = dcflatten_wrap(t, nv, elens);   % natural boundary condition
%     cprintf('r', 'var(mu) %e,llcr error %e\n', norm( abs(fmeshmiur(t, vf{1}, z) ))^2/nf, norm(fllcr( fr2c(z) )-fllcr( fr2c(x{1}) )) );
%     fprintf( '%s blending with w %s: %dBrokTri, dcferr %.3e, flaterr %.3e\n', opt.metric, num2str(w(iw,:),'[%.3f %.3f]'), nbrokentris, dcferr, flaterr );

    if ~isempty( anchId )
        dirs = cellfun( @(y) fNormalize( [1 -1]*fr2c(y(anchId(1:2),:)) ), x );
        dir = prod( dirs.^w(iw,:) );

        rot = dir./fNormalize([1 -1]*fr2c(z(anchId(1:2),:)));
%         pos = cellfun( @(y) sum(fr2c(y(anchId(1:2),:)))/2, x )*w(iw,:)';
%         z = fc2r( (fr2c(z)-fr2c(sum(z(anchId(1:2),:))/2)).*rot + pos );
        
%         pos = cellfun( @(y) fr2c(y(anchId(1),:)), x )*w(iw,:)';
        pos = fr2c(x{1}(anchId(1),:));
        z = fc2r( (fr2c(z)-fr2c(z(anchId(1),:))).*rot + pos );
    end
    
    Z{iw} = z;

    %% viz
    if showmorph
        set(h(1), 'Vertices', z);
        set(h(2), 'XData', z(anchId,1), 'YData', z(anchId,2));
        drawnow; pause(pt);
    end
end

if numel(Z) == 1, Z = Z{1}; end