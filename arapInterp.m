function Z = arapInterp(x, t, w, opt)

if size(w, 2)>2
    [x, w] = fixInterpInput4ClassicAlg(x, w);
end


% [x, t, anchIds, colors] = genMorphData( 'spirals', 1 );
% opt.anchors = anchIds; opt.hf = figuredocked; opt.pt=0.3;
% w = (0:50)' / 50;
% w = [w 1-w];

if nargin<4
    opt = struct('hf', -1);
end

if isfield(opt, 'anchors') && ~isempty(opt.anchors)
    anchId = opt.anchors(1);
else
    anchId = 1;
end

showmorph = isfield(opt, 'hf') && ishandle(opt.hf);

if showmorph
    if isfield(opt, 'fMC')
        fMeshColoring = opt.fMC;
    else
%         fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'flat', 'EdgeAlpha', 0.1, 'FaceVertexCData', cdata);
        fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'flat', 'EdgeColor', 'none', 'FaceVertexCData', cdata);
    end
    fPlotMarkers = @(x) plot( x(:,1), x(:,2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    if ~isfield( opt, 'pause' ); opt.pause = 0; end
    pt = opt.pause;
    
    cla; h = drawmesh(t, x{1}); title('ARAP');  hold on; 
    fMeshColoring(h, x{1}(:,2));
    h(2) = fPlotMarkers( x{1}(anchId,:) );
end


%%
nv = size(x{1}, 1);
nf = size(t, 1);

%%
V2E = [0 1 -1; -1 0 1];
% fjacob = @(x, y) ( (V2E*x)\(V2E*y) )';
fjacob = @(x, y) (V2E*x)\(V2E*y);
fc2mat = @(x) [real(x) imag(x); -imag(x) real(x)];
% ffun2faceCell = @(f, t, x, y) cellfun(f, fXPerFace(t,x), fXPerFace(t,y), 'UniformOutput', false);
% fmeshjacobs = @(t, x, y) ffun2faceCell(fjacob, t, x, y);

% H = sparse(nf*2, nv);
% for i=1:nf
%     % E_x*B = E_y, B = inv(E_x)*E_y
%     H(i*2+(-1:0),:) = (V2E*x{1}(t(i,:),:))\sparse( [1 1 2 2], t(i,[2 3 3 1]), [1 -1 1 -1], 2, nv );
%     
% %     H(i*2+(-1:0),t(i,:)) = (V2E*x{1}(t(i,:),:))\V2E;
% end

fComputeM = @(x) (x(:,[4 2 2 3 1 1])-x(:,[6 6 4 5 5 3]))./((dot(x(:,1:3),x(:,4:6),2)-dot(x(:,[1 3 5]), x(:,[6 2 4]),2))*[1 -1 1 -1 1 -1]);

xPerFace = reshape(x{1}(t',:)',6,[])';
M = fComputeM(xPerFace);
H = sparse( [repmat((1:nf)'*2-1,1,3) repmat((1:nf)'*2,1,3)], [t t], M, nf*2, nv );


%% polar decocmposotion, with loops
% R = zeros(nf, 1);
% S = cell(nf, 1);
% 
% for i=1:nf
%     jacob = fjacob(x{1}(t(i, :),:), x{2}(t(i, :),:));
% 
%     [u,s,v] = svd(jacob);
%     assert( det(u*v)>=0 );
%     S{i} = v*s*v';
%     rot = u*v';
%     R(i) = complex(rot(1,1), rot(1,2));
% end
% S = cell2mat( cellfun(@(m) reshape(m', 1, []), S, 'UniformOutput', false) );

%% polar decocmposotion, vectorized
fNormalize = @(x) x./abs(x);
fBestRots = @(a) fNormalize( complex(a(:,1)+a(:,4), a(:,2)-a(:,3)) );

fc2r = @(x) [real(x) imag(x)];
fa2r = @(x) [cos(x) sin(x)];
fMultiRots2Affs = @(R, A) [dot(R, A(:,[1 3]), 2) ...
                           dot(R, A(:,[2 4]), 2) ...
                           dot([-R(:,2) R(:,1)], A(:,[1 3]), 2) ...
                           dot([-R(:,2) R(:,1)], A(:,[2 4]), 2)];
fDevCompRotsByAffs = @(R, A) fMultiRots2Affs(fc2r(conj(R)), A);
fMulM2xPF = @(M, x) [dot(M(:,1:3), x(:,1:2:5),2) dot(M(:,1:3), x(:,2:2:6),2) dot(M(:,4:6), x(:,1:2:5),2) dot(M(:,4:6), x(:,2:2:6),2)];

%
affs = fMulM2xPF(M, reshape(x{2}(t',:)',6,[])');
R = fBestRots(affs); S = fDevCompRotsByAffs(R, affs);


a = angle(R);
useFix = 1;
VV2T = sparse( t, t(:, [2 3 1]), repmat(1:nf, 1, 3), nv, nv );

%%
if useFix == 1  % new fix
    fR2C = @(x) complex(x(:,1), x(:,2));
    x1 = fR2C(x{1});
    eInF = x1(t(:,[3 1 2])) - x1(t(:,[2 3 1]));
    clear x1;
    A = signedAreas(x{1}, t)*4;
    E = sparse( repmat(1:nf, 3, 1)', t, conj(1i*eInF)./[A A A], nf, nv );
    fz = E*fR2C(x{2});

    [eIi, eIj] = find(VV2T&VV2T');
    eI = unique(sort([eIi eIj], 2), 'rows');
    neI = size(eI,1);
    fVV2T = @(e) VV2T( sub2ind([nv nv], e(:,1), e(:,2)) );

    angleDif = angle( fz(fVV2T(eI)) ./ fz(fVV2T(eI(:,[2 1]))) );
    MV2Evec = sparse( repmat((1:neI)', 1, 2), [fVV2T(eI) fVV2T(eI(:,[2 1]))], repmat([1 -1], neI, 1), neI, nf );
    a = [MV2Evec; sparse(1, 1, 1, 1, nf)]\[angleDif; imag(log(fz(1)))];
elseif useFix == 2 % Baxter's rotation fix
    ct = 1;
    % MM = sparse( repmat((1:nf)', 1, 3), t, 1, nf, nv );
    % [~, tRings] = nRingInTriMesh( MM, 1 );

    ttlist = full( VV2T( sub2ind([nv nv], t(:, [2 3 1]), t(:, [1 2 3])) ) );

    tflag = false(1,nf+1);
    ttlist(ttlist==0) = nf+1;
    tflag(ct) = 1;
    t2p = [ct];
    while ~isempty(t2p)
        ct = t2p(1);
        tt = ttlist(t2p(1),:);
    %     ct = t2p(end);
    %     tt = ttlist(t2p(end),:);
        tt = tt( tt<=nf & ~tflag(tt) );

        if ~isempty(tt)
            ca = a(ct);
            i1 = a(tt)-ca>pi;
            i2 = a(tt)-ca<-pi;
            if any(i1)
                a(tt(i1)) = a(tt(i1)) - ceil( (a(tt(i1))-ca)/2/pi - 1e-1 ) * 2*pi;
            end
            if any(i2)
                a(tt(i2)) = a(tt(i2)) + ceil( (a(tt(i2))-ca)/2/-pi - 1e-1 ) * 2*pi;
            end
            tflag(tt) = true;
        end

        t2p = [t2p(2:end) tt];
    end
end

% R = exp(1i*a);
% set(h(1), 'FaceColor', 'flat', 'EdgeAlpha', 0.1, 'FaceVertexCData', a); colorbar;

%%
I = setdiff(1:nv, anchId);

A = signedAreas(x{1}, t);
AA = sparse(1:2*nf, 1:2*nf, reshape(sqrt([A A]'), 1, []), 2*nf, 2*nf);

H = AA*H;
H1 = H(:,anchId);    H2 = H(:,I);



%%
nw = size(w,1);
Z = cell(nw,1);
for iw = 1:nw
    wt = w(iw,2);

%     MA = zeros(nf*2,2);
%     for i=1:nf
%         MA(i*2+(-1:0),:) = fc2mat(R(i)^wt)*( (1-wt)*eye(2)+wt*S{i} );
%     end

    if useFix
        MA = fMultiRots2Affs( fa2r(a*wt), repmat((1-wt)*[1 0 0 1], nf, 1)+wt*S );
    else
        MA = fMultiRots2Affs( fc2r(R.^wt), repmat((1-wt)*[1 0 0 1], nf, 1)+wt*S );
% %         S2 = cell2mat( arrayfun(@(i) reshape(reshape(S(i,:), 2, 2)^wt, 1, []), 1:size(S,1), 'UniformOutput', false)' );
%         S2 = zeros(nf, 4);
%         for i=1:nf
%             [v, s] = eig( reshape(S(i,:), 2, 2) );
%             s = diag(s);
%             s1 = s(1)*wt+1-wt;
%             S2(i, :) = reshape( v*diag([s1 (mean(s)^wt*2-s1)])*v', 1, 4 );
%         end
%         MA = fMultiRots2Affs( fc2r(R.^wt), S2 );        
    end
        
    MA = AA*reshape(MA', 2, [])';

%     zAnchors = [1-wt wt]*[x{1}(anchId,:); x{2}(anchId,:)]; % interp anchId pos
    zAnchors = x{1}(anchId,:); % interp anchId pos
%     zAnchros = zAnchros + repmat( x{1}(anchId,:)-zAnchros(1,:), size(zAnchros, 1), 1 ); % fixed anchId1 at keyframe 1

    % cvx_begin quiet
    %     variables z2(nv,2);
    %     expressions MB(nf*2,2);
    %     for i=1:nf
    %         MB(i*2+(-1:0),:) = (V2E*x(t(i,:),:))\(V2E*z2(t(i,:),:));
    %     end
    % 
    %     minimize( norm(MA-MB, 'fro') );
    % 
    %     subject to
    %         z2(1,:) == z1;
    % cvx_end

    % cvx_begin quiet
    %     variables z3(nv,2);
    %     minimize( norm(H*z3-MA, 'fro') );
    %     subject to
    %         z3(1,:) == z1;
    % cvx_end

    %%
    z([anchId I],:) = [zAnchors; H2 \ (MA-H1*zAnchors)];
    
%     % explicit least square
%     A1 = H'*H; b1 = H'*MA;
%     norm( z([anchId I],:) - [zAnchros; A1(I,I)\(b1(I,:)-A1(I,anchId)*zAnchros)] )
    
    
    fprintf('LS error = %e\n', norm( H*z-MA )^2/sum(A));
    % assert( norm(z-z2,'fro') < 1e-7 );

    Z{iw} = z;
    if showmorph
        set(h(1), 'Vertices', z);
        set(h(2), 'XData', z(anchId,1), 'YData', z(anchId,2));
        set(h(1), 'CData', sqrt( sum(reshape( (H*z-MA)', 4, [] ).^2)/sum(A) ) );
        drawnow; pause(pt);
    end
end


if numel(Z)==1, Z = Z{1}; end
