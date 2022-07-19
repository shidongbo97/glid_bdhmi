function [z, allStats] = meshSLIM(xref, t, P2PVtxIds, P2PDst, z, nIter, p2p_weight, energy_type, energy_param)
% z: initilization

if isempty(P2PVtxIds), allStats=zeros(1,8); return; end

if nargin < 7, p2p_weight = 1e9; end
if nargin < 8, energy_type = 'SymmDirichlet'; end
if nargin < 9, energy_param = 1; end

nv = numel(z);
nf = size(t, 1);

if size(xref,1)==nv
    xref = xref(t(:,2:3)) - xref(t(:,1)); % implicit the x1 = 0, for each triangle
end

% faceElens = sqrt( meshFaceEdgeLen2s(xref, t) );
% faceAngles = meshAnglesFromFaceEdgeLen2(faceElens.^2);
% xref = [faceElens(:,3) faceElens(:,2).*exp(1i*faceAngles(:,1))];

e = xref*[-1 0 1; 1 -1 0];


%% initilization
isometric_energyies = [ "SymmDirichlet", "ExpSD", "AMIPS", "SARAP", "HOOK", "ARAP", "BARAP", "BCONF"];

findStringC = @(s, names) find(strcmpi(s, names), 1) - 1;
mexoption = struct('energy_type', findStringC(energy_type, isometric_energyies), ...
                   'hessian_projection', "NP", 'energy_param', energy_param, 'verbose', 0);
               
Areas = (real(xref(:,1)).*imag(xref(:,2)) - imag(xref(:,1)).*real(xref(:,2)))/2;
D2 = -1i/4*(xref*[1 0 -1; -1 1 0])./Areas;
D2t = D2.';
D = sparse(repmat(1:nf,3,1)', t, D2);

P2Plhs = 2*p2p_weight*sparse(P2PVtxIds, P2PVtxIds, 1, nv, nv);
P2Prhs = 2*p2p_weight*sparse(P2PVtxIds, 1, P2PDst, nv, 1);


% fIsoEnergyFromFzGz = @(fz, gz) meshIsometricEnergy(D2, fz, gz, Areas, energy_type, energy_param);
fIsoEnergyFromFzGz = @(fz, gz) meshIsometricEnergyC(fz, gz, D2t, Areas, mexoption);

fDeformEnergy = @(z) fIsoEnergyFromFzGz(conj(D*conj(z)), D*z) + p2p_weight*norm(z(P2PVtxIds)-P2PDst)^2;

en = fDeformEnergy(z);
ls_beta = 0.5;
ls_alpha = 0.2;

allStats = zeros(nIter+1, 8); % statistics
allStats(1, [5 7 8]) = [0 norm(z(P2PVtxIds)-P2PDst)^2 en];

for it=1:nIter
    tt = tic;
    
    %% local
    fz = D*conj(z);
    gz = D*z;

    U = abs(fz).*abs(gz) + fz.*gz;
    U(abs(U)<1e-20) = 1;  % avoid division by 0
    U = U./abs(U);
    U2 = U.^2;
    S = abs([fz gz])*[1 1; 1 -1];
    
    switch energy_type
    case 'ARAP'
        Sw = [1 1];
    case 'SymmDirichlet'
        Sw = (S.^-2+1).*(S.^-1+1);
    case 'Exp_SymmDirichlet'
        Sw = (S.^-2+1).*(S.^-1+1).*exp(energy_param*(S.^2+S.^-2))*2*energy_param;
    otherwise
        assert('energy %s not implemented for SLIM', energy_type);
    end
    
    Sv = Sw*[1 -1; 1 1]/2;  % sv1 > sv2
    e2 = Sv(:,1).*e + Sv(:,2).*U2.*conj(e);
    w2 = real( e(:,[2 3 1]).*conj(e2(:,[3 1 2])) )./Areas;
    L2 = sparse( t(:,[2 3 1 3 1 2]), t(:,[3 1 2 2 3 1]), [w2 w2]/2, nv, nv );
    L2 = spdiags(-sum(L2,2), 0, L2);

    Rots = conj(fz)./abs(fz);
    
    %% global poisson
    b = accumarray( reshape(t,[],1), reshape(e2*1i.*Rots, [], 1) );
    
    z2 = (L2+P2Plhs) \ (b+P2Prhs);
    dz = z2 - z;

%     G = (L2+P2Plhs)*z - (b+P2Prhs);
    [~, g] = fIsoEnergyFromFzGz(conj(fz), gz);
    G = reshape(sparse([t t+nv]', 1, g, nv*2, 1), [], 2)*[1; 1i] + P2Plhs*z-P2Prhs;
   
    
    %% orientation preservation
    ls_t = min( maxtForPositiveArea( fz, gz, D*conj(dz), D*dz )*0.9, 1 );
    
    %% line search energy decreasing
    fMyFun = @(t) fDeformEnergy( dz*t + z );
    normdz = norm(dz);
    dgdotfz = dot( [real(G); imag(G)], [real(dz); imag(dz)] );
    fQPEstim = @(t) en+ls_alpha*t*dgdotfz;

    e_new = fMyFun(ls_t);
    while ls_t*normdz>1e-12 && e_new > fQPEstim(ls_t)
        ls_t = ls_t*ls_beta;
        e_new = fMyFun(ls_t);
    end
    en = e_new;

    fprintf('it: %3d, t: %.3e, en: %.3e\n', it, ls_t, en);
    
    %% update
    z = dz*ls_t + z;

    %% stats
    allStats(it+1, [5 7 8]) = [toc(tt)*1000 norm(z(P2PVtxIds)-P2PDst)^2 en];
    
    if norm(G)<1e-4, break; end
    if norm(dz)*ls_t<1e-10, break; end
end

allStats = allStats(1:it+1, :);
allStats(:,7:8) = allStats(:,7:8)/sum(Areas);

% fprintf('%dits: mean runtime: %.3e\n', nIter, mean(allStats(2:end,5)));
  