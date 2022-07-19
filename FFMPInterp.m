function Z = FFMPInterp(x, t, w, opt)

if size(w, 2)>2
    [x, w] = fixInterpInput4ClassicAlg(x, w);
end

if nargin<4
    opt = struct('hf', -1);
end

if isfield(opt, 'anchors') && ~isempty(opt.anchors)
    anchId = double(opt.anchors);
else
    anchId = 1;
end

showmorph = isfield(opt, 'hf') && ishandle(opt.hf);

if showmorph
    if isfield(opt, 'fMC')
        fMeshColoring = opt.fMC;
    else
        fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceAlpha', 0.8, 'FaceVertexCData', cdata);
%         fMeshColoring = @(h, cdata) set(h, 'FaceColor', 'w', 'EdgeColor', [1 1 1]*0.4, 'LineSmoothing', 'on');
    end

    fPlotMarkers = @(x) plot( x(:,1), x(:,2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
%     fPlotMarkers = @(x) [];
%     fShowTriangle = @(x) patch(x(:,1), x(:,2), zeros(size(x,1),1), 'facecolor', 'none', 'linewidth', 3, 'edgecolor', 'g');
    fShowTriangle = @(x) plot(x([1:end 1],1), x([1:end 1],2), 'linewidth', 3, 'color', 'g');

    if ~isfield( opt, 'pause' ); opt.pause = 0; end
    pt = opt.pause;

    cla; h = drawmesh(t, x{1}); title('FFMP'); hold on; 
    fMeshColoring(h(1), x{1}(:,2));
end

%%
nv = size(x{1}, 1);
nf = size(t, 1);

fc2mat = @(x) [real(x) imag(x); -imag(x) real(x)];
fc2mats = @(x) [real(x) imag(x) -imag(x) real(x)];
flininterp = @(alpha, x, y) (1-alpha)*x+alpha*y;

fNormalize = @(x) x./abs(x);
fBestRots = @(a) fNormalize( complex(a(:,1)+a(:,4), a(:,2)-a(:,3)) );

%%
% V2E = [0 1 -1; -1 0 1];

% G = sparse( repmat( (1:nf*2)', 1, 2 ), reshape(t(:,[2 1 3 1])', 2, [])', [ones(nf*2, 1) -ones(nf*2, 1)], nf*2, nv );
G = sparse( repmat((1:nf)'*2,1,4) + repmat([-1 -1 0 0], nf, 1 ), t(:,[2 1 3 1]), repmat([1 -1 1 -1], nf, 1 ), nf*2, nv );
GtG = G'*G;

[Q, t12] = quadInTriMesh(t, nv);
t1 = t12(:,1); t2 = t12(:,2);

nq = size(Q, 1);

% Di = @(x, i) G(i*2+(-1:0),:)*x;
Dit = @(x, i) G(reshape([i*2-1; i*2],[],1), :)*x;
Qij = @(x, i, j) Dit(x,j)'\Dit(x,i)';

%%
% Rs = zeros(nf, 1); Ss = cell(nf, 1);
% Rt = zeros(nf, 1); St = cell(nf, 1);
% for i=1:nf
%     [u1,s1,v1] = svd( Dit(x{1}, i)' );
%     Ss{i} = v1*s1*v1';
%     rot = u1*v1';
%     Rs(i) = complex(rot(1,1), rot(1,2));
% 
%     [u2,s2,v2] = svd( Dit(x{2},i)' );
%     St{i} = v2*s2*v2';
%     rot = u2*v2';
%     Rt(i) = complex(rot(1,1), rot(1,2));
% end
% 
% % Ss2 = cell2mat( cellfun(@(m) reshape(m', 1, []), Ss, 'UniformOutput', false) );
% % St2 = cell2mat( cellfun(@(m) reshape(m', 1, []), St, 'UniformOutput', false) );
% Rs2 = Rs; Rt2 = Rt; Ss2 = Ss; St2 = St;

%%
fInvSymm2x2Mats = @(A) [A(:,4) -A(:,2) -A(:,2) A(:,1)]./repmat( A(:,1).*A(:,4)-A(:,2).*A(:,3), 1, 4 );

fMul2x2Mats = @(A, B) [dot(A(:,1:2), B(:,[1 3]), 2) ...
                       dot(A(:,1:2), B(:,[2 4]), 2) ...
                       dot(A(:,3:4), B(:,[1 3]), 2) ...
                       dot(A(:,3:4), B(:,[2 4]), 2)];

fMultiRots2Affs = @(R, A) [dot(R, A(:,[1 3]), 2) ...
                           dot(R, A(:,[2 4]), 2) ...
                           dot([-R(:,2) R(:,1)], A(:,[1 3]), 2) ...
                           dot([-R(:,2) R(:,1)], A(:,[2 4]), 2)];

fc2r = @(x) [real(x) imag(x)];

%% divided rotation (represented as complex number) by affine map
fDevCompRotsByAffs = @(R, A) fMultiRots2Affs(fc2r(conj(R)), A);

%%
affs = reshape( Dit(x{1},1:nf)', 4, [] )'; affs = affs(:, [1 3 2 4]); 
Rs = fBestRots( affs ); Ss = fDevCompRotsByAffs(Rs, affs);

afft = reshape( Dit(x{2},1:nf)', 4, [] )'; afft = afft(:, [1 3 2 4]); 
Rt = fBestRots( afft ); St = fDevCompRotsByAffs(Rt, afft);


vfi = anchId;

% set a triangle with anchId(1) to be a anchor triangle
% tfi = 1;
[tfi, ~] = find(t==anchId(1), 1);
GtG(vfi,:) = sparse(1:numel(vfi), vfi, 1, numel(vfi), nv);

%%
nw = size(w,1);
Z = cell(nw,1);
for iw = 1:nw
    wt = w(iw,2);

%     H = sparse(nq*2, nf*2);
%     % for i=1:nq
%     %     H(i*2+(-1:0), [t12(i,1)*2+(-1:0) t12(i,2)*2+(-1:0)]) = [-speye(2) Qij(x, t12(i,1), t12(i,2))];
%     % end
% 
%     for i=1:nq
%         ct1 = t12(i,1); ct2 = t12(i,2);
%         st1 = flininterp(wt, Ss2{ct1}, St2{ct1});
%         st2 = flininterp(wt, Ss2{ct2}, St2{ct2});
%         Q12 = st2\fc2mat( flininterp(wt, Rs2(ct2)\Rs2(ct1), Rt2(ct2)\Rt2(ct1)) )*st1;
% 
% %         assert( norm(Dit(x,ct2)'*Qij(x, ct1, ct2)-Dit(x,ct1)', 'fro')<1e-10 );
% %         assert( norm(Dit(y,ct2)'*Qij(y, ct1, ct2)-Dit(y,ct1)', 'fro')<1e-10 );
% 
% %         assert( norm(Q12 - Qij(x, ct1, ct2), 'fro')<1e-10 );
% %         assert( norm(Q12'*Dit(x, ct2) - Dit(x, ct1))<1e-10) ;
% 
%         H(i*2+(-1:0), [ct1*2+(-1:0) ct2*2+(-1:0)]) = [-speye(2) Q12'];
%     end
    
   
    Q12 = fMul2x2Mats( fInvSymm2x2Mats(flininterp(wt,Ss(t2,:),St(t2,:))), fc2mats(flininterp(wt, Rs(t2).\Rs(t1), Rt(t2).\Rt(t1))) );
    Q12 = fMul2x2Mats( Q12, flininterp(wt,Ss(t1,:),St(t1,:)) );
    
    H = sparse([(1:nq)*2-1; (1:nq)*2]', [t1*2-1 t1*2], -ones(nq*2, 1), nq*2, nf*2);
    H = H+sparse([(1:nq)'*2-1 (1:nq)'*2-1 (1:nq)'*2 (1:nq)'*2], [t2*2-1 t2*2 t2*2-1 t2*2], Q12(:,[1 3 2 4]), nq*2, nf*2);
%     H( sub2ind([nq nf]*2, [(1:nq)'*2-1 (1:nq)'*2-1 (1:nq)'*2 (1:nq)'*2], [t2*2-1 t2*2 t2*2-1 t2*2]) ) = Q12(:,[1 3 2 4]);
        
    %%
    H = H'*H;

%     Dfix = fc2mat( Rs(tfi)^(1-wt)*Rt(tfi)^wt )*flininterp(wt, Ss{tfi}, St{tfi});
    Dfix = fc2mat( Rs(tfi)^(1-wt)*Rt(tfi)^wt )*reshape(flininterp(wt, Ss(tfi,:), St(tfi,:))', 2, 2);
    
    % Dt = zeros(nf*2, 2);
    % Dt(tfi*2+(-1:0), :) = Dfix;
    % tvi2 = [tvi*2-1 tvi*2];

    H(tfi*2+(-1:0), :) = sparse( 1:2, tfi*2+(-1:0), [1 1], 2, nf*2 );
    rhs = zeros(nf*2,2);
    rhs( tfi*2+(-1:0),: ) = Dfix';

    Dt = H\rhs;

    %%
    Dt = G'*Dt;

    Dt(vfi,:) = flininterp(wt, x{1}(vfi,:), x{2}(vfi,:));
    z = GtG\Dt;

    
    
    if numel(anchId)>1
        fc2r = @(x) [real(x) imag(x)];
        fr2c = @(x) complex(x(:,1), x(:,2));
        fNormalize = @(x) x./abs(x);
        dirs = cellfun( @(y) fNormalize( [1 -1]*fr2c(y(anchId(1:2),:)) ), x );
        dir = prod( dirs(1:2).^w(iw,:) );

        rot = dir./fNormalize([1 -1]*fr2c(z(anchId(1:2),:)));
        
        % interp mid point between anchors as global translation
%         pos = cellfun( @(y) mean(fr2c(y(anchId(1:2),:))), x )*w(iw,:)';
%         z = fc2r( (fr2c(z)-fr2c(sum(z(anchId(1:2),:))/2)).*rot + pos );

        % interp anchor1 pos as global translation
        pos = fr2c(x{1}(anchId(1),:));        
%         pos = cellfun( @(y) fr2c(y(anchId(1),:)), x )*w(iw,:)';
        z = fc2r( (fr2c(z)-fr2c(z(anchId(1),:))).*rot + pos );
    end

    Z{iw} = z;
    
    if showmorph
        set(h(1), 'Vertices', z);

        delete(h(2:end));
        h(2:3) = [fShowTriangle(z(t(tfi,:),:)); fPlotMarkers( z(vfi,:) )];

        drawnow; pause(pt);
    end
end

if numel(Z)==1, Z = Z{1}; end
