fC2R = @(x) [real(x) imag(x)];
fR2C = @(x) complex(x(:,1), x(:,2));

t = T;
x = X;

nf = size(t, 1);
nv = numel(x);


e1 = [2 3 1];   e2 = [3 1 2];
VE = full( sparse( [1:3 1:3], [e1 e2], -[1 1 1 -1 -1 -1]) );
VE2 = VE(1:2, :);

e = x(t(:,e2)) - x(t(:,e1));

% fComputeM0 = @(x) cellfun(@(tri) (VE2*x(tri,:))\VE2, mat2cell(t, ones(nf,1), 3), 'UniformOutput', false); % version 2, use left division
% fFlatMats = @(x) cellfun(@(m) reshape(m', 1, []), x, 'UniformOutput', false);
% M0 = cell2mat(fFlatMats(fComputeM0(fC2R(x))));

fComputeM = @(x) (x(:,[5 4 4 2 1 1])-x(:,[6 6 5 3 3 2]))./((dot(x(:,[1 4 2]),x(:,[5 3 6]),2)-dot(x(:,1:3), x(:,[6 4 5]),2))*[1 -1 1 -1 1 -1]);
M1 = fComputeM( fC2R(x(t)) );

% Jacobian is concatenated in x directions
fComputeJacobR = @(M, z) [M(:,[1 1]).*z(t(:,1), :)+M(:,[2 2]).*z(t(:,2), :)+M(:,[3 3]).*z(t(:,3), :) M(:,[4 4]).*z(t(:,1), :)+M(:,[5 5]).*z(t(:,2), :)+M(:,[6 6]).*z(t(:,3), :)];
fComputeJacob = @(M, z) [sum(M(:,1:3).*z(t),2) sum(M(:,4:6).*z(t),2)];

%% functors
fNormalize = @(x) x./abs(x);
fBestRots = @(a) fNormalize( complex(a(:,1)+a(:,4), a(:,2)-a(:,3)) );

%% initilization to identity
z = x;

figuredocked; h = drawmesh(t, z); hold on;
set(h, 'FaceColor', 'w', 'edgealpha', 0.1);
hp = plot(P2PCurrentPositions, 'ro');
hp0 = plot(z(P2PVtxIds), 'ko');


I = setdiff(1:nv, P2PVtxIds);
Areas = signedAreas(x, t);


energy_type = 'ISO';
switch energy_type
case 'ARAP'
    fIsoEnergy = @(sigs) dot( Areas, sum( (sigs-1).^2, 2) );
case 'ISO'
    fIsoEnergy = @(sigs) dot( Areas, sum(sigs.^2+sigs.^-2, 2) );
case 'EISO'
    fIsoEnergy = @(sigs) dot( Areas, exp( sum(sigs.^2+sigs.^-2, 2)*energy_parameter ) );
otherwise
    warning('not supported energy: %s!', energy_type);
end

lambda = 1e9;
P2Plhs = 2*lambda*sparse(P2PVtxIds, P2PVtxIds, 1, nv, nv);
P2Prhs = 2*lambda*sparse(P2PVtxIds, 1, P2PCurrentPositions, nv, 1);

fComputeSigmas = @(z) abs( fComputeJacobR(M1, fC2R(z))*[1 1; -1i 1i; 1i 1i; 1 -1]/2 )*[1 1; 1 -1];
fDeformEnergy = @(z) fIsoEnergy( fComputeSigmas(z) ) + lambda*norm(z(P2PVtxIds)-P2PCurrentPositions)^2;

en = fDeformEnergy(z);

for it=1:100
    %% local
    affs = fComputeJacobR(M1, fC2R(z));
    Rots = fBestRots(affs);

    %% compute differentials
    fz = affs*[1; -1i; 1i; 1]/2;
    gz = affs*[1; 1i; 1i; -1]/2;
    S = abs([fz gz])*[1 1; 1 -1];
%     S(:,2) = abs(S(:,2)); % S>0, since locally injective
    
    %% SLIM
    U2 = fz.*gz;
    U2(abs(U2)<1e-10) = 1;  % avoid division by 0
    U2 = U2./abs(U2);
    Sw = (S.^-2+1).*(S.^-1+1);
    Sv = Sw*[1 -1; 1 1]/2;  % sv1 > sv2
    e2 = Sv(:,1).*e + Sv(:,2).*U2.*conj(e);
%     e2 = e;
    w2 = real( e(:,[2 3 1]).*conj(e2(:,[3 1 2])) )./Areas;
    L2 = sparse( t(:,[2 3 1 3 1 2]), t(:,[3 1 2 2 3 1]), [w2 w2]/2, nv, nv );
    L2 = spdiags(-sum(L2,2), 0, L2);

    %% global poisson
%     b = sparse( t(:, [e1 e2]), ones(nf, 6), [-ones(nf,3) ones(nf,3)].*repmat(e.*repmat(Rots,1,3).*ctgsx, 1, 2), nv, 1 );
%     b = accumarray( reshape(t,[],1), reshape(e*1i.*Rots, [], 1) );
    b = accumarray( reshape(t,[],1), reshape(e2*1i.*Rots, [], 1) );
%     z2(I) = L2(I, I)\(b(I,:) - L2(I,P2PVtxIds)*P2PCurrentPositions);
    z2 = (L2+P2Plhs) \ (b+P2Prhs);

    %% orientation preservation
%     ls_t = 1;    
    ls_t = min( min( maxtForPositiveArea( z(t)*VE(:,1:2), z2(t)*VE(:,1:2) ) )*0.9, 1 );
    

    %% line search energy decreasing
    fMyFun = @(t) fDeformEnergy( z2*t + z*(1-t) );
%     normdz = norm(z-z2);
    e_new = fMyFun(ls_t);
    while e_new > en
        ls_t = ls_t/2;
        e_new = fMyFun(ls_t);
    end
    en = e_new;

    fprintf('it: %3d, t: %.3e, en: %.3e\n', it, ls_t, en);

    %% update
    z = z2*ls_t + z*(1-ls_t);
    
    %%
    title( sprintf('iter %d', it) );
    set(h, 'Vertices', fC2R(z));
    set(hp0, 'XData', real(z(P2PVtxIds)), 'YData', imag(z(P2PVtxIds)));
    drawnow;
    pause(0.002);


    %% testing
%     bCorner = (Sv(:,1).*e + Sv(:,2).*U2.*conj(e))*1i.*Rots;
%     fC2Rm = @(x) [real(x) -imag(x); imag(x) real(x)];
% 
%     for i=1:nf
%         A = reshape(affs(i,:),2,2)';
%         [u, s, v] = svd(A);
%         s0 = diag(s)';
%         assert(s0*[1;-1]>=0);
%         
%         u = fC2Rm(U(i));
%         
%         sw = (s0.^-2+1).*(s0.^-1+1);
%         
% %         u = eye(2); sw(:) = 1;
%         
%         sv = sw*[1 -1; 1 1]/2;  % sv1 > sv2        
%         W = diag(sw.^0.5)*u';
%         
%         WTW = sv(1)*eye(2) - sv(2)*u*[1 0; 0 -1]*u'; % = W'*W
%         
%         E = VE*x(t(i,:));
%         Er = fC2R(E*1i);
%         Er0 = fC2R(E);
%         L1 = Er*W'*W*Er'/Areas(i);
% 
% %         Er*u*[1 0; 0 -1]*u'*Er'
% %         Er0*u*[1 0; 0 -1]*u'*Er0'
% %         
% %         real( E*conj(U(i)) * conj(U(i))*E.' )
% %         real( conj(U(i))^2*E*E.' )
%         
% %         L2 = real(sv(1)*E*E' + sv(2)*conj(U(i))^2*E*E.')/Areas(i)
%         L2 = real( E*(sv(1)*E' + sv(2)*conj(U(i))^2*E.') )/Areas(i);
% 
%         E2 = sv(1)*E + sv(2)*U(i)^2*conj(E);
%         L3 = real(E*E2') / Areas(i);
%         
%         w = real( E([2 3 1]).*conj(E2([3 1 2])) )/Areas(i);
%         L4 = full( sparse( [2 3 1 3 1 2], [3 1 2 2 3 1], [w; w].', 3, 3 ) );
%         
%         b1 = Er*W'*W*fC2Rm(Rots(i))';
% %         b4 = E*1i.*sv(1)*Rots(i) + conj(E)*1i*sv(2)*U(i)^2*Rots(i)
%         b4 = (E*sv(1) + conj(E)*sv(2)*U(i)^2)*1i*Rots(i);
%         
%         [bCorner(i,:)-fR2C(b1).'  w2(i,:)-w.']
% %         u2 = U(i);
% %         Ru = [real(u2) -imag(u2); imag(u2) real(u2)];
% %         A*A' - Ru*s.^2*Ru'
% % 
% %         [S(i,:) s0']
%     end
end

