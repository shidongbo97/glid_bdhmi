function [z, allStats] = meshNewton(xref, t, P2PVtxIds, P2PDst, y, nIter, p2p_weight, energy_type, energy_param, hession_proj)

fR2C = @(x) complex(x(:,1), x(:,2));

if ~exist('hession_proj', 'var'), hession_proj = 'SymmDirichlet'; end
if ~exist('energy_param', 'var'), energy_param = 1; end


if isreal(xref), xref = fR2C(xref); end
if isreal(y), y = fR2C(y); end
if isreal(P2PDst), P2PDst = fR2C(P2PDst); end

nf = size(t, 1);
nv = size(y, 1);

if isempty(P2PVtxIds)
    [~,P2PVtxIds] = min(abs(y-mean(y)));
    P2PDst = y(P2PVtxIds);
end

%%
if size(xref,1)==nv
    xref = xref(t(:,2:3)) - xref(t(:,1)); % implicit the x1 = 0, for each triangle
end


%% initilization
isometric_energyies = [ "SymmDirichlet", "ExpSD", "AMIPS", "SARAP", "HOOK", "ARAP", "BARAP", "BCONF"];
hessian_projections = [ "NP", "KP", "FP4", "FP6", "CM" ];



findStringC = @(s, names) find(strcmpi(s, names), 1) - 1;
mexoption = struct('energy_type', findStringC(energy_type, isometric_energyies), ...
                   'hessian_projection', findStringC(hession_proj, hessian_projections), ...
                   'energy_param', energy_param, 'verbose', 0);

z = y;

Areas = (real(xref(:,1)).*imag(xref(:,2)) - imag(xref(:,1)).*real(xref(:,2)))/2;
D2 = -1i/4*(xref*[1 0 -1; -1 1 0])./Areas;
D2t = D2.';
D = sparse(repmat(1:nf,3,1)', t, D2);


% fIsoEnergyFromFzGz = @(fz, gz) meshIsometricEnergy(D2, fz, gz, Areas, energy_type, energy_param);
fIsoEnergyFromFzGz = @(fz, gz) meshIsometricEnergyC(fz, gz, D2t, Areas, mexoption);

fDeformEnergy = @(z) fIsoEnergyFromFzGz(conj(D*conj(z)), D*z) + p2p_weight*norm(z(P2PVtxIds)-P2PDst)^2;

%% initialization, get sparse matrix pattern
[xmesh, ymesh] = meshgrid(1:6, 1:6);
t2 = [t t+nv]';
Mi = t2(xmesh(:), :);
Mj = t2(ymesh(:), :);

% H = sparse(Mi, Mj, 1, nv*2, nv*2);  % only pattern is needed
L = laplacian(y, t, 'uniform');
H = [L L; L L];

% nonzero indices of the matrix
Hnonzeros0 = zeros(nnz(H),1);
P2PxId = uint64( [P2PVtxIds P2PVtxIds+nv] );
idxDiagH = ij2nzIdxs(H, P2PxId, P2PxId);
Hnonzeros0(idxDiagH) = p2p_weight*2;
nzidx = ij2nzIdxs(H, uint64(Mi), uint64(Mj));

G0 = zeros(nv*2,1);

solver = splsolver(H, 'ldlt');

allStats = zeros(nIter+1, 8); % statistics
allStats(1, [5 7 8]) = [0 norm(z(P2PVtxIds)-P2PDst)^2 fDeformEnergy(z)];

%% main loop
g2GIdx = uint64(t2);
for it=1:nIter
    tt = tic;

    fz = conj(D*conj(z)); % equivalent but faster than conj(D)*z;
    gz = D*z;
    
    assert( all( abs(fz)>abs(gz) ) );

    [en, g, hs] = fIsoEnergyFromFzGz(fz, gz);

    en = en + p2p_weight*norm(z(P2PVtxIds)-P2PDst)^2;
    dp2p = z(P2PVtxIds) - P2PDst;
    G0(P2PxId) = [real(dp2p); imag(dp2p)]*p2p_weight*2;
    G = myaccumarray(g2GIdx, g, G0);
    Hnonzeros = myaccumarray( nzidx, hs, Hnonzeros0 );
    
    %% Newton
    dz = solver.refactor_solve(Hnonzeros, -G);
    dz = fR2C( reshape(dz, [], 2) );
    
    %% orientation preservation
    ls_t = min( maxtForPositiveArea( fz, gz, conj(D*conj(dz)), D*dz )*0.9, 1 );
    
    %% line search energy decreasing
    fMyFun = @(t) fDeformEnergy( dz*t + z );
    normdz = norm(dz);

    dgdotfz = dot( G, [real(dz); imag(dz)] );
    
    ls_alpha = 0.2; ls_beta = 0.5;
    fQPEstim = @(t) en+ls_alpha*t*dgdotfz;

    e_new = fMyFun(ls_t);
    while ls_t*normdz>1e-12 && e_new > fQPEstim(ls_t)
        ls_t = ls_t*ls_beta;
        e_new = fMyFun(ls_t);
    end
    en = e_new;
    
    fprintf('\nit: %3d, en: %.3e, runtime: %.3fs, ls: %.2e', it, en, toc(tt), ls_t*normdz);
    assert( all( signedAreas(dz*ls_t + z, t)>0 ) );
    
    %% update
    z = dz*ls_t + z;
    
    %% stats
    allStats(it+1, [5 7 8]) = [toc(tt)*1000 norm(z(P2PVtxIds)-P2PDst)^2 en];
    
    
    if norm(G)<1e-4, break; end
    if norm(dz)*ls_t<1e-10, break; end
end

allStats = allStats(1:it+1, :);
allStats(:,7:8) = allStats(:,7:8)/sum(Areas);

