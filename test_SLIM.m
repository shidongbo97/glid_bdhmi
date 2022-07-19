n = numel(vv);
phipsy = [vv vv*0];

P2Pdst = P2PCurrentPositions;
% P2Pdst = fComposeHM(C(P2PVtxIds,:), phipsy);

fComposeHM = @(C, phipsy) C*phipsy(:,1)+conj(C*phipsy(:,2));
fNormalize = @(x) x./abs(x);

lambda = 1e5;
CC = C(P2PVtxIds,:);

% P2Plhs = 2*lambda*[CC conj(CC)]'*[CC conj(CC)];
% P2Prhs = 2*lambda*[CC conj(CC)]'*P2Pdst;

P2Plhs = [CC conj(CC)]'*[CC conj(CC)];
P2Prhs = [CC conj(CC)]'*P2Pdst;


fIsoEnergy = @(sigmas) sum( sum( sigmas.^2 + sigmas.^-2 ) );
fDeformEnergy = @(pp) fIsoEnergy( abs(D2*pp)*[1 1; 1 -1] ) + lambda*norm( fComposeHM(C(P2PVtxIds,:), pp)-P2Pdst )^2;
en = fDeformEnergy(phipsy);


z = fComposeHM(C, phipsy);
figuredocked; h = drawmesh(T, z); hold on;
set(h, 'FaceColor', 'w', 'edgealpha', 0.1);
hp = plot(P2Pdst, 'ro');
hp0 = plot(z(P2PVtxIds), 'ko');


for it=1:200
    fzgz = D2*phipsy;
    Rots = fNormalize(fzgz(:,1));
    S = abs(fzgz)*[1 1; 1 -1];
%     min(S(:,2))
    
    %% SLIM
%     U = abs(prod(fzgz,2)) + prod(fzgz,2);
%     U(abs(U)<1e-10) = 1;  % avoid division by 0
%     U = U./abs(U);
%     U2 = U.^2;
    U2 = prod(fzgz,2);
    U2(abs(U2)<1e-10) = 1;
    U2 = U2./abs(U2);

    Sw = (S.^-2+1).*(S.^-1+1);
    alphas = Sw*[1;1]/2;
    betas = Sw*[-1;1]/2;
    
    La = D2'*( alphas.*D2 );
    Lb = D2'*( -(betas.*U2).*conj(D2) );
    L = [La Lb; Lb' conj(La)];

    b = [D2'*(Rots.*alphas); D2.'*(-Rots.*betas.*conj(U2))];

%     fGradIso = @(fzgz, fzgz2) [D2'*(fzgz(:,1).*(1-((fzgz2*[1;-1]).^-3).*(fzgz2*[1;3])) ); conj(D2'*(fzgz(:,2).*(1+((fzgz2*[1;-1]).^-3).*(fzgz2*[3;1]))))];
%     [ L*[phipsy(:,1); conj(phipsy(:,2))]-b   fGradIso(fzgz, abs(fzgz).^2) ]

    
    %% global poisson
    phipsy2 = reshape( (L+2*lambda*P2Plhs) \ (b+2*lambda*P2Prhs), [], 2 );
    phipsy2(:,2) = conj(phipsy2(:,2));

    %% orientation preservation
    ls_t = 1;
    dfzgz = D2*phipsy2 - fzgz;
    maxts = maxtForPhiPsyBat(fzgz(:,1), fzgz(:,2), dfzgz(:,1), dfzgz(:,2));
    ls_t = min(1, min(maxts)*0.9);

    %% line search energy decreasing
    fMyFun = @(t) fDeformEnergy( phipsy2*t + phipsy*(1-t) );
    e_new = fMyFun(ls_t);
    while e_new > en
        ls_t = ls_t/2;
        e_new = fMyFun(ls_t);
    end
    en = e_new;

    stepnorm = ls_t*norm(phipsy2-phipsy);
    fprintf('it: %3d, t: %.3e, stepsize: %.3e, en: %.3e\n', it, ls_t, stepnorm, en);

    %% update
    phipsy = phipsy2*ls_t + phipsy*(1-ls_t);

    %%
    z = gather( fComposeHM(C, phipsy) );
    title( sprintf('iter %d', it) );
    set(h, 'Vertices', fC2R(z));
    set(hp0, 'XData', real(z(P2PVtxIds)), 'YData', imag(z(P2PVtxIds)));
    drawnow;
    pause(0.002);
    
    if stepnorm<1e-16, break; end
end
