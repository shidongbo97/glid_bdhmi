function Z = arapLGInterp(x, t, w, opt)

if size(w, 2)>2
    [x, w] = fixInterpInput4ClassicAlg(x, w);
end

if nargin<4
    opt = struct('hf', -1);
end

if isfield(opt, 'anchors') && ~isempty(opt.anchors)
    anchId = double(opt.anchors);
else
    anchId = diameter(x{1});
end

showmorph = isfield(opt, 'hf') && ishandle(opt.hf);
if isfield(opt, 'NumIter');  maxNumIter = opt.NumIter; else maxNumIter = 20; end

if showmorph
    if isfield(opt, 'fMC');
        fMeshColoring = opt.fMC;
    else
        fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'flat', 'FaceAlpha', 0.8, 'FaceVertexCData', cdata);
%         fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'w', 'EdgeColor', [1 1 1]*0.4, 'LineSmoothing', 'on');
    end
    fPlotMarkers = @(x) plot( x(:,1), x(:,2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');

    if ~isfield( opt, 'pause' ); opt.pause = 0; end
    pt = opt.pause;

    cla; h = drawmesh(t, x{1}); title('ARAP LG');  hold on; 
    fMeshColoring(h(1), x{1}(:,2)); % axis on;
    h(2) = fPlotMarkers( x{1}(anchId,:) );
end

%%
nv = size(x{1}, 1);
nf = size(t, 1);
B = findBoundary(x{1}, t);

e1 = [2 3 1];   e2 = [3 1 2];
VE = full( sparse( [1:3 1:3], [e1 e2], -[1 1 1 -1 -1 -1]) );
VE2 = VE(1:2, :);

fr2c = @(x) complex(x(:,1), x(:,2));
fc2r = @(x) [real(x) imag(x)];

A1 = signedAreas(x{1}, t);
A2 = signedAreas(x{2}, t);

%% build LSCM/ARAP related matrices
MA = sparse( [t t+nv], [nv+t(:, [2 3 1]) t(:,[2 3 1])], [ones(nf, 3) -ones(nf, 3)], nv*2, nv*2 );
MA = (MA+MA')/2;

L1 = cotLaplace(x{1}, t);
L2 = cotLaplace(x{2}, t);

fEdgeVecsC = @(x) x(t(:,e2)) - x(t(:,e1));
ex = fEdgeVecsC(fr2c(x{1}));
ey = fEdgeVecsC(fr2c(x{2}));

ctgsx = cot(meshAngles(x{1}, t));
ctgsy = cot(meshAngles(x{2}, t));

fComputeM = @(x) cellfun(@(tri) (VE2*x(tri,:))\VE2, mat2cell(t, ones(nf,1), 3), 'UniformOutput', false); % version 2, use left division
fFlatMats = @(x) cellfun(@(m) reshape(m', 1, []), x, 'UniformOutput', false);

M1 = cell2mat(fFlatMats(fComputeM(x{1})));
M2 = cell2mat(fFlatMats(fComputeM(x{2})));
fComputeJacob = @(M, z) [M(:,[1 1]).*z(t(:,1), :)+M(:,[2 2]).*z(t(:,2), :)+M(:,[3 3]).*z(t(:,3), :) M(:,[4 4]).*z(t(:,1), :)+M(:,[5 5]).*z(t(:,2), :)+M(:,[6 6]).*z(t(:,3), :)];

%% functors
fNormalize = @(x) x./abs(x);
fBestRot = @(a) fNormalize( complex(a(1,1)+a(2,2), a(1,2)-a(2,1)) );
fBestRots = @(a) fNormalize( complex(a(:,1)+a(:,4), a(:,2)-a(:,3)) );

fDiffAff2RotAreaWtC = @(A, a, r) norm(repmat(sqrt(A),1,4).*(a-[real(r) imag(r) -imag(r) real(r)]), 'fro')^2;

%%
R1 = ones(nf,1);
R2 = ones(nf,1);

I = setdiff(1:nv, anchId);

nw = size(w,1);
Z = cell(nw,1);
z = x{1};
for iw = 1:nw
    wt = w(iw,2);
    zAnchor = (1-wt)*x{1}(anchId,:) + wt*x{2}(anchId,:);
    L = (1-wt)*L1 + wt*L2;
    z(anchId,:) = zAnchor;

    %% use ASAP as initlization
    % conformal laplacian
    CL = blkdiag( L, L ) - MA*2;

    CL([anchId anchId+nv], :) = sparse(1:4, [anchId anchId+nv], ones(1,4), 4, nv*2);
    b = zeros(nv*2, 1);
    
    b( [anchId anchId+nv] ) = zAnchor(:);

    z = CL\b;
    z = reshape( z, [], 2 );

    E = ones(1, 4)*inf;
    for it=1:maxNumIter
       %% updating rotation matrices
        Eold = E;

        aff1 = fComputeJacob(M1, z);
        aff2 = fComputeJacob(M2, z);
        E(1) = fDiffAff2RotAreaWtC(A1, aff1, R1);
        E(3) = fDiffAff2RotAreaWtC(A2, aff2, R2);

        R1 = fBestRots(aff1);
        R2 = fBestRots(aff2);

        E(2) = fDiffAff2RotAreaWtC(A1, aff1, R1);
        E(4) = fDiffAff2RotAreaWtC(A2, aff2, R2);

        %%
        signchar = '-+';
        fEn2Str = @(e, eold) sprintf('%.3e(%s)', e, signchar((e>eold+eps)+1));
        avgEold = (1-wt)*Eold(1:2) + wt*Eold(3:4);
        avgE = (1-wt)*E(1:2) + wt*E(3:4);
        if ~mod(it-1, ceil(5e4/nv))
            fprintf('Local/Global step %2d, Ex= %s:%s, Ey=%s:%s, Ex+Ey=%s:%s\n', it, fEn2Str(E(1), Eold(1)), fEn2Str(E(2), E(1)), fEn2Str(E(3), Eold(3)), fEn2Str(E(4), E(3)), fEn2Str(avgE(1), avgEold(1)), fEn2Str(avgE(2), avgE(1)) );
        end
        if norm(avgE(2))<1e-5 || norm(avgE - avgEold)<1e-5
            break;
        end

       %% global poisson
        b1 = sparse( t(:, [e1 e2]), ones(nf, 6), [-ones(nf,3) ones(nf,3)].*repmat(ex.*repmat(R1,1,3).*ctgsx, 1, 2), nv, 1 );
        b2 = sparse( t(:, [e1 e2]), ones(nf, 6), [-ones(nf,3) ones(nf,3)].*repmat(ey.*repmat(R2,1,3).*ctgsy, 1, 2), nv, 1 );

        b = fc2r( full( (1-wt)*b1 + wt*b2 ) );
        
    %     L = cotLaplace(x, t);
        z(I, :) = L(I, I)\(b(I,:) - L(I,anchId)*zAnchor);
    end

    Z{iw} = z;

    if showmorph
        set(h(1), 'Vertices', z);
        set(h(2), 'XData', z(anchId,1), 'YData', z(anchId,2));
        drawnow; pause(pt);
    end
end


if numel(Z)==1, Z = Z{1}; end