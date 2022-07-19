function [z, err] = GBDHInterp(x, t, wts, opt)

fC2R = @(x) [real(x) imag(x)];
fR2C = @(x) complex(x(:,1), x(:,2));
fNormalize = @(x) x./abs(x);


if isreal(x{1}), x = cellfun(fR2C, x, 'UniformOutput', false); end

x = cell2mat(x);

if nargin<4, opt=struct(); end

if isfield(opt, 'anchors') && ~isempty(opt.anchors)
    anchId = opt.anchors;
else
    anchId = diameter( fC2R(x(:,1)) );
end


showmorph = isfield(opt, 'hf') && ishandle(opt.hf);
if showmorph
    cla; h = drawmesh(t, x(:,1)); hold on;  %title('GBDH');
end

x0 = x(:,1);
nv = size(x0, 1);
nf = size(t, 1);
eInF = x0(t(:,[3 1 2])) - x0(t(:,[2 3 1]));
A = signedAreas(x0, t)*4;

E = sparse( repmat(1:nf, 3, 1)', t, conj(1i*eInF)./[A A A], nf, nv );
F = conj(E);

fz = E*x;
fzbar = conj(E)*x;

%% g = smarter version of log(fz)
e = [t(:) reshape(t(:, [2 3 1]), [], 1)];
e = unique( sort(e, 2), 'rows' );

vv2t = sparse(t, t(:, [2 3 1]), repmat( (1:nf)', 1, 3 ), nv, nv);
% vv2t(B, B) = 0; % will cause problem for ear triangles
[eIi, eIj] = find(vv2t&vv2t');
eI = unique(sort([eIi eIj], 2), 'rows');
neI = size(eI,1);
fVV2T = @(e) vv2t( sub2ind([nv nv], e(:,1), e(:,2)) );

fzr = fz(fVV2T(eI),:) ./ fz(fVV2T(eI(:,[2 1])),:);


gdif = complex(log(abs(fzr)), angle(fzr));
MV2Evec = sparse( repmat((1:neI)', 1, 2), [fVV2T(eI) fVV2T(eI(:,[2 1]))], repmat([1 -1], neI, 1), neI, nf );

logfz1 = log(fz(1,:));
% if isfield(opt, 'logfz1Ref'), logfz1 = opt.logfz1Ref - round( imag(logfz1-logfz1Ref)/2/pi)*2i*pi; end
g = [MV2Evec; sparse(1, 1, 1, 1, nf)]\[gdif; logfz1];



fFractionNorm = @(x) norm( x-round(x) );
if norm(MV2Evec*g-gdif)>1e-5
    fprintf('branching error in log(fz): %f: %f\n', norm(MV2Evec*g-gdif), fFractionNorm( (g-log(fz))/2i/pi ));
end


k = abs(fzbar./fz);
kmax = max(k, [], 2);

W = sparse(1:2*nf, 1:2*nf, [A; A].^0.5);
%%
nwt = size(wts, 1);
z = cell(nwt, 1);
for i = 1:nwt
    wt = wts(i, :);
%     fzt = fz.^wt;
    fzt = exp(g*wt');
    etat0 = (fz.*fzbar)*wt';
    k0 = abs(etat0./fzt.^2);
    sscale = min( min(1, kmax./k0) );
    
    sscale = 1;
    etat = etat0*sscale;
    fzbart = etat./fzt;
    
    
%     eVecs = repmat(fzt, 1, 3).*eInF + conj( repmat(conj(fzbart), 1, 3).*eInF );

%%    
    z{i} = ( [W*[E; F]; sparse(1,1,1,1,nv)])\([W*[fzt; fzbart]; 0]);

%     Lcot = laplacian(x, t);
%     L1 = [E; conj(E)]'*W'*W*[E; conj(E)];
%     L2 = [E1; conj(E1)]'*[E1; conj(E1)];
%     fprintf('Least square error = %f, cvx status = %s\n', norm(W*([E; F]*fR2C(z{i})-[fzt; fzbart]), 'fro'), cvx_status);

%     k1 = abs( (conj(E)*fR2C(z{i}))./ (E*fR2C(z{i})) );
%     err = norm(W*([E; F]*fR2C(z{i})-[fzt; fzbart]), 'fro')^2/sum(A);
%     errmax = max( abs([E; F]*fR2C(z{i})-[fzt; fzbart]) );
%     fprintf('t=%.2f Least square error = %.3e(max: %.3e), max_k = %.3e \n', wt, err, errmax, max(k1) );
%     err(2) = errmax;


    if ~isempty( anchId )
        dirs = fNormalize( [1 -1]*x(anchId(1:2),:) );
        dir = prod( dirs.^wt );

        rot = dir./fNormalize([1 -1]*z{i}(anchId(1:2),:));
%         pos = cellfun( @(y) sum(fr2c(y(anchId(1:2),:)))/2, x )*w(iw,:)';
%         z = fc2r( (fr2c(z)-fr2c(sum(z(anchId(1:2),:))/2)).*rot + pos );

%         pos = cellfun( @(y) fr2c(y(anchId(1),:)), x )*w(iw,:)';
        pos = x(anchId(1), 1);
        z{i} = (z{i} - z{i}(anchId(1)))*rot + pos;
    end



    if showmorph
%         set(h, 'Vertices', fC2R(z{i}), 'FaceColor', 'flat', 'EdgeColor', 'none',  'CData', sqrt( sum(reshape( abs(W*([E; F]*fR2C(z{i})-[fzt; fzbart])).^2, [], 2 ), 2)/sum(A) ) );
        set(h, 'Vertices', fC2R(z{i}), 'facecolor', 'interp', 'CData', imag(x(:,1)), 'EdgeAlpha', 0.02); colormap jet;
        
        drawmeshEx(t, z{i});
        drawnow;
    end

    z{i} = fC2R( z{i} );
end

if nwt==1 && iscell(z), z = z{1}; end
