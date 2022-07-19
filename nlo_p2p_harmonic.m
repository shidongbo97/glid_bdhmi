function [phipsyIters, allStats] = nlo_p2p_harmonic(invM_AQP, D2, C2, bP2P, softP2P, lambda, phipsyIters, energy_parameter, AQPKappa, nIter, solver, energy_type, ...
    nextSampleInSameCage, hessianSampleRate, fillDistanceSegments, v, E2, L)

linesearchLIM = nargin>12;
linesearchLIM2 = true;

cageSizes = gather( int32( cellfun(@numel, v) ) );
vv = cat(1, v{:});

if ~softP2P, lambda = 0; end

n = size(D2,2);
numEnergySamples = size(D2, 1);

enEvalsPerKernel = 10;

hessSampleStride = ceil(1/hessianSampleRate);
hessian_samples = 1: hessSampleStride :ceil(numEnergySamples-hessSampleStride/2);
if hasGPUComputing, hessian_samples = gpuArray(hessian_samples); end
hessianSampleRate = numel(hessian_samples)/numEnergySamples;

cageOffsets = cumsum([0 cageSizes(1) cageSizes(2:end)-2]);  % remove first 2 vertex in each hole


c_iso_energy_names = {'SymmDirichlet', 'Exp_SymmDirichlet', 'ARAP', 'AMIPS' };
isometric_energy_type = find(strcmpi(c_iso_energy_names, energy_type), 1) - 1;
if isempty(isometric_energy_type), isometric_energy_type = -1; end

if numel(v)==1, v = v{1}; end

optimization_methods = {'GD', 'AQP', 'Newton', 'Newton_SPDH', 'Newton_SPDH_FullEig'};
CUDA_SOLVER_NAMES = {'cuAQP single', 'cuGD single', 'cuNewton single', 'cuNewton_SPDH single', 'cuNewton_SPDH_FullEig single', ...
               'cuAQP', 'cuGD', 'cuNewton', 'cuNewton_SPDH', 'cuNewton_SPDH_FullEig'};
switch solver
    case CUDA_SOLVER_NAMES
        singlePrecision = numel(solver)>7 && strcmpi(solver(end-6:end), ' single');
        if singlePrecision
            solver = solver(1:end-7); 
            fillDistanceSegments = single(fillDistanceSegments);
            v = single(v); L = single(L); E2 = single(E2);
        end
            
        optmethod = find(strcmpi(solver(3:end), optimization_methods), 1) - 1;
        
        if optmethod==1 % AQP
            assert(size(invM_AQP,1)==n*2, 'invM for AQP has to be set before calling mex!');
        end

        params = struct('hessian_samples', int32(hessian_samples-1), 'isometric_energy_type', isometric_energy_type, 'isometric_energy_power', energy_parameter, 'aqp_kappa', AQPKappa, 'nIter', nIter,  ...
                        'sample_spacings_half', fillDistanceSegments, 'v', vv, 'E2', E2, 'L', L, 'nextSampleInSameCage', int32(nextSampleInSameCage-1), ...
                        'LS_energy_eval_per_kernel', enEvalsPerKernel, 'solver', optmethod, 'linearSolvePref', 0, 'deltaFixSPDH', 1e-15, 'reportIterationStats', 1);
                    % linearSolvePref PREFER_CHOLESKY = 0, PREFER_LU = 1, FORCE_CHOLESKY = 2     
                    % deltaFixSPDH, is relative to p2p_weight
        
        if singlePrecision
            [phipsyIters, allStats] = cuHarmonic( single(invM_AQP), single(D2), single(C2), single(bP2P), lambda, single(phipsyIters), cageOffsets, params); 
        else
            [phipsyIters, allStats] = cuHarmonic( invM_AQP, D2, C2, bP2P, lambda, double(phipsyIters), cageOffsets, params);
        end    

        if size(allStats,1)>2
            allStats = allStats';
        else
            allStats = [zeros(1, 7) allStats];
        end
        return;
end

mu = energy_parameter;

fDistortionGradImp = @(fzalpha1, gzalpha2, evec) 2*[D2'*(fzalpha1.*evec); conj(D2'*(gzalpha2.*evec))]; 

%% energy functions for different energy types
switch energy_type
    case 'ARAP'
        fGradIso = @(fzgzb, fzgzb2) 4*[D2'*(fzgzb(:,1).*(1-1./abs(fzgzb(:,1)))); conj(D2'*fzgzb(:,2))]; 
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( 2*sum(fzgz2, 2)-4*fzgz2(:,1).^0.5 + 2 );
        linesearchLIM = false;
        linesearchLIM2 = false;
    case 'BARAP'
        fGradIso = @(fzgz, fzgz2) 2*[D2'*(fzgz(:,1).*(2-2./abs(fzgz(:,1)) + mu - mu*diff(fzgz2,1,2).^-2));
                                conj(D2'*(fzgz(:,2).*(2                   - mu + mu*diff(fzgz2,1,2).^-2)))]; 
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( 2*sum(fzgz2, 2)-4*fzgz2(:,1).^0.5 + 2 - mu./diff(fzgz2,1,2) );
    case 'SymmDirichlet'
        if mu==1
            fIsometryicEnergyFzGz2 = @(fzgz2) sum( sum(fzgz2, 2).*(1+diff(fzgz2, 1, 2).^-2) );
            fGradIso = @(fzgz, fzgz2) 2*[D2'*(fzgz(:,1).*(1-((fzgz2*[1;-1]).^-3).*(fzgz2*[1;3])) ); conj(D2'*(fzgz(:,2).*(1+((fzgz2*[1;-1]).^-3).*(fzgz2*[3;1]))))];
        else
            fIsometryicEnergyFzGz2 = @(fzgz2) sum( abs(sum(fzgz2, 2).*(1+diff(fzgz2, 1, 2).^-2) ).^mu );
            fGradIso = @(fzgz, fzgz2) fDistortionGradImp( fzgz(:,1).*(1+(diff(fzgz2,1,2).^-3).*(fzgz2*[1;3])), ...
                                                          fzgz(:,2).*(1-(diff(fzgz2,1,2).^-3).*(fzgz2*[3;1])), ...
                                                          mu*abs(sum(fzgz2,2).*(1+diff(fzgz2,1,2).^-2) ).^(mu-1) );
        end
    case 'Exp_SymmDirichlet'
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( exp( mu*abs(sum(fzgz2, 2).*(1+diff(fzgz2, 1, 2).^-2)) ) );
        fGradIso = @(fzgz, fzgz2) fDistortionGradImp( fzgz(:,1).*(1+(diff(fzgz2,1,2).^-3).*(fzgz2*[1;3])), ...
                                                      fzgz(:,2).*(1-(diff(fzgz2,1,2).^-3).*(fzgz2*[3;1])), ...
                                                      mu*exp( mu*abs(sum(fzgz2,2).*(1+diff(fzgz2,1,2).^-2)) ) );
    case 'AMIPS'
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( exp( (mu*2*sum(fzgz2,2)+1)./-diff(fzgz2,1,2) - diff(fzgz2,1,2) ) );
        fGradIso = @(fzgz, fzgz2) fDistortionGradImp( fzgz(:,1).* (1-(4*mu*fzgz2(:,2)+1).*diff(fzgz2,1,2).^-2), ...
                                                      fzgz(:,2).*-(1-(4*mu*fzgz2(:,1)+1).*diff(fzgz2,1,2).^-2), ...
                                                      exp( (mu*2*sum(fzgz2,2)+1)./-diff(fzgz2,1,2) - diff(fzgz2,1,2) ) );
    case 'Beta'
        assert(false, 'Todo: to be fixed');
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( sum( sqrt(fzgz2), 2 ).^2 + diff( sqrt(fzgz2), 1, 2 ).^-2 );
        fGradBETA_IMP = @(fzgz, x, y) 2*[D2'*(fzgz(:,1).*(1 + x.^-1.*(y-(x-y).^-3))); 
                                    conj(D2'*(fzgz(:,2).*(1 + y.^-1.*(x+(x-y).^-3))))];
        fGradIso = @(fzgz, fzgz2) fGradBETA_IMP(fzgz, sqrt(fzgz2(:,1)), sqrt(fzgz2(:,2)));

    case {'SymmARAP', 'NeoHookean'}
        assert(false, 'Todo: to be implimented');
    otherwise
        assert(false);
end

% fP2PEnergyPhiPsy2 = @(phipsyb) real([phipsyb; 1]'*matEnP2P*[phipsyb; 1]);
fP2PEnergyPhiPsy = @(fg) norm(fg(:,1)+conj(fg(:,2)) - bP2P)^2;

% matEnP2P = [C2 conj(C2) -bP2P]'*[C2 conj(C2) -bP2P];
matGradP2P = 2*[C2 conj(C2)]'*[C2 conj(C2) -bP2P];
fGradP2P = @(phi, psy) matGradP2P*[phi; conj(psy); ones(1,size(phi,2))];


fC2Rm = @(x) [real(x) -imag(x); imag(x) real(x)];
fR2Cv = @(x) complex(x(1:end/2), x(end/2+1:end));
fC2Rv = @(x) [real(x); imag(x)];

CtC = [C2 conj(C2)]'*[C2 conj(C2)];
CtCr = fC2Rm(CtC);
CtCr(1:n*3, n*3+(1:n)) = -CtCr(1:n*3, n*3+(1:n));
CtCr(n*3+(1:n), 1:n*3) = -CtCr(n*3+(1:n), 1:n*3);


ls_beta = 0.5;
ls_alpha = 0.2;

if ~softP2P
    fMyEnergy = @(fzgz2, fg) fIsometryicEnergyFzGz2(fzgz2);
    fMyGrad = @(fzgz, fzgz2, phi, psy) fGradIso(fzgz, fzgz2);
else
    fMyEnergy = @(fzgz2, fg) fIsometryicEnergyFzGz2(fzgz2) + fP2PEnergyPhiPsy(fg)*lambda;
    fMyGrad = @(fzgz, fzgz2, phi, psy) fGradIso(fzgz, fzgz2) + fGradP2P(phi, psy)*lambda;
end


%% nullspace, reduce problem size by removing redundant vars
if ~iscell(v)
    N = speye(2*n, 2*n-1);
    nHoles = 0;
else
    cageSz = numel(v{1});
    isFreeVar = true(1, n*2);
    nHoles = numel(v)-1;
    isFreeVar([n+cageSz 2*n+(1-nHoles:0)]) = false;
    N = sparse([find(isFreeVar) 2*n+(1-nHoles:0)], [1:sum(isFreeVar) n+(1-nHoles:0)],  1, 2*n, sum(isFreeVar));
end

if hasGPUComputing && verLessThan('matlab', '9.2'), N = full(N); end

%%
if strcmpi(solver, 'Newton_ConformalMap')
    N = N(:, 1:n-nHoles);
end

% for real variables
Nr = blkdiag(N, N);
Nr(end-numel(v)+2:end,:) = -Nr(end-numel(v)+2:end,:);

switch solver
    case CUDA_SOLVER_NAMES % processed before, pass
    
    case 'LBFGS'
        %% rest pose isometric hessian, can only represented in real numbers
%         D2r = [real(D2) -imag(D2)];
%         h = 16*blkdiag( D2r'*D2r, fC2Rm(D2'*D2) );
%         RIRI2RRII = [1:n n*2+(1:n) n+(1:n) n*3+(1:n)];
%         h  = h(RIRI2RRII, RIRI2RRII);

        %% using Dirichlet energy hessian as initial hessian, already reordered by real/imag of phi psy
%         h = 2*blkdiag(D2'*D2, conj(D2'*D2));

        %% identity initial hessian for isometric energy
        h = eye(2*n)*2;
        M = h + 2*lambda*CtC;

        %% for LBFGS, a PSD initial hessian is need for the algorithm to converge, so remove the 
        % global DOF of harmonic map from the problem
        invM = N*inv(N'*M*N)*N';

        LBFGS_K = 5; % parameter k

        fzgzAll = D2*phipsyIters;
        gIterAll = reshape(phipsyIters, [], size(phipsyIters,2)/2)*0;
        for i=1:size(phipsyIters,2)/2
            j = i*2+(-1:0);
            gIterAll(:, i) = fGradIso(fzgzAll(:,j), abs(fzgzAll(:,j)).^2) + fGradP2P(phipsyIters(:, j(1)), phipsyIters(:, j(2)))*lambda;
        end
        
        
        allStats = zeros(nIter+1, 8);
        allStats(1, 8) = gather( fMyEnergy( abs(fzgzAll(:,1:2)).^2, C2*phipsyIters(:,1:2)) );
        for it=1:nIter
            tic

            %% iteration
            fzgz = fzgzAll(:, 1:2);
            ppIter = reshape(phipsyIters, n*2, []);
            ppIter(n+1:end,:) = conj( ppIter(n+1:end,:) );
            fgP2P = C2*phipsyIters(:, 1:2);
            
            dpp = lbfgs_iter(invM, gIterAll, ppIter);
            dppdotg = dot( [real(dpp); imag(dpp)], [real(gIterAll(:,1)); imag(gIterAll(:,1))] );
            dpp = reshape(dpp, [], 2);
            
            normdpp = norm(dpp);            
            dpp(:,2) = conj(dpp(:,2));
            
            dfzgz = D2*dpp;
            dfgP2P = C2*dpp;
            fMyFun = @(t) fMyEnergy(abs(fzgz+t*dfzgz).^2, fgP2P+t*dfgP2P);

            maxts = arrayfun(@maxtForPhiPsy, fzgz(:,1), fzgz(:,2), dfzgz(:,1), dfzgz(:,2));
            ls_t = min(1, min(maxts)*0.8); % faster than min( [maxts;1] )
            
            e = fMyEnergy(abs(fzgz).^2, fgP2P);
            fQPEstim = @(t) e+ls_alpha*t*dppdotg;
            e_new = fMyFun(ls_t);
            while ls_t*normdpp>1e-12 && e_new > fQPEstim(ls_t)
                ls_t = ls_t/2;
                e_new = fMyFun(ls_t);
            end

            if linesearchLIM
                ls_t = lineSearchLocallyInjectiveHarmonicMap(phipsyIters(:,1:2), dpp, fzgz(:,1:2), dfzgz(:,1:2), ls_t, fillDistanceSegments, v, E2, L, nextSampleInSameCage);
                e_new = fMyFun(ls_t);
            end

            phipsyIters = [phipsyIters(:,1:2)+ls_t*dpp phipsyIters(:,1:min(end,2*(LBFGS_K-1)))];

            %% update for next iteration
            fzgz = D2*phipsyIters(:,1:2);

            gIter = fGradIso(fzgz, abs(fzgz).^2) + fGradP2P(phipsyIters(:, 1), phipsyIters(:, 2))*lambda;
            gIterAll = [gIter gIterAll(:, 1:min(end, (LBFGS_K-1)))];
            fzgzAll = [fzgz fzgzAll(:, 1:min(end, 2*(LBFGS_K-1)))];

            allStats(it+1, [5 8]) = [toc*1000 gather( real(e_new))];
%             fprintf('LBFGS it %d: runtime: %.3es, energy: %.3e\n', it, allStats(it,5)/1000, allStats(it,8));
        end
        
        fprintf('LBFGS %diterations: mean runtime: %.3ems\n', it, mean(allStats(2:end,5)));

    case {'bemAQP', 'bemSLIM', 'Newton', 'Newton_SPDH', 'Newton_ConformalMap', 'Newton_SPDH_FullEig', 'Gradient Descent'}
        phipsyIters = double(phipsyIters);

        phi = phipsyIters(:, 1);
        psy = phipsyIters(:, 2);
        
        fzgz0 = D2*[phi psy];   % gz: conj(fzb)
        fzgz2 = abs(fzgz0).^2;  % abs2([fz conj(fzb)])
        
        e = fMyEnergy(fzgz2, C2*[phi psy]);
        flagCanAccel = false; % for acceleration in AQP
        
        % statistics to be in consistant with that returned from cuda solvers
        allStats = zeros(nIter+1, 8);
        allStats(1, [7 8]) = gather( [fP2PEnergyPhiPsy(C2*[phi psy])*lambda e] );

        for it=1:nIter
            tic;
            fgP2P = C2*[phi psy];
            
            if AQPKappa>1 && flagCanAccel
                theta = (1-sqrt(1/AQPKappa))/(1+sqrt(1/AQPKappa));

                dpp = phipsyIters(:, 1:2)-phipsyIters(:, 3:4);
                dfzgz = D2*dpp;
                
                if linesearchLIM2
                    maxts = maxtForPhiPsyBat(fzgz0(:,1), fzgz0(:,2), dfzgz(:,1), dfzgz(:,2));
                    theta = min(theta, min(maxts)*0.8);
                end

                ls_t = theta;

                if linesearchLIM
                    ls_t = lineSearchLocallyInjectiveHarmonicMap(phipsyIters(:,1:2), dpp, fzgz0, dfzgz, ls_t, fillDistanceSegments, v, E2, L, nextSampleInSameCage);
                end
                
                phi = phi + dpp(:,1)*ls_t;
                psy = psy + dpp(:,2)*ls_t;

                fzgz0 = fzgz0 + ls_t*dfzgz;
                fzgz2 = abs(fzgz0).^2;
                
                fgP2P = C2*[phi psy];
                e = fMyEnergy(fzgz2, fgP2P);
            end

            g = fMyGrad(fzgz0, fzgz2, phi, psy);

            if strncmpi(solver, 'Newton', 6)
                SPDHessian = strcmp(solver, 'Newton_SPDH');
                [~, ~, h] = harmonicMapIsometryicEnergy(D2(hessian_samples,:), phi, psy, SPDHessian, energy_type, mu);

                h = h/hessianSampleRate;

                if strcmpi(solver, 'Newton_SPDH_FullEig')
                    [eigD, eigE] = eig( tril(h) + tril(h)' - diag(diag(h)) );
                    eigE(eigE<0) = 0;
                    h = eigD*eigE*eigD';
                end

                M = h + 2*lambda*CtCr;
            end


            %% solve
            if strcmp(solver, 'Gradient Descent')
                dpp = -g;
            else
                if ~softP2P % AQP only
                    assert( strcmp(solver, 'bemAQP'), 'hard P2P is only supported for AQP' );
                    dzP2P = bP2P - ( C2*phi + conj(C2*psy) );
                    dpp = invM_AQP*[-g; dzP2P];
                else
                    if strncmpi(solver, 'Newton', 6)
                        g(n+1:n*2) = conj( g(n+1:n*2) );
                        dpp = fR2Cv( Nr*( (Nr'*M*Nr)\(Nr'*fC2Rv(-g)) ) );
                    elseif strcmp(solver, 'bemSLIM')
                        U2 = prod(fzgz0,2);
                        U2(abs(U2)<1e-10) = 1;
                        U2 = U2./abs(U2);

                        S = abs(fzgz0)*[1 1; 1 -1];

                        switch energy_type
                        case 'ARAP'
                            Sw = [1 1];
                        case 'SymmDirichlet'
                            Sw = (S.^-2+1).*(S.^-1+1);
                        case 'Exp_SymmDirichlet' 
                            Sw = (S.^-2+1).*(S.^-1+1).*exp(energy_parameter*(S.^2+S.^-2))*2*energy_parameter;
                        otherwise
                            assert(false,'energy %s not implemented for bemSLIM', energy_type);
                        end

                        alphas = Sw*[1;1]/2;
                        betas = Sw*[-1;1]/2;

                        La = D2'*( alphas.*D2 );
                        Lb = D2'*( -(betas.*U2).*conj(D2) );
                        H = [La Lb; Lb' conj(La)];
                        dpp = -N*( (N'*(H+2*lambda*CtC)*N)\(N'*g) );
                    else %AQP soft P2P
                        dpp = invM_AQP*(-g);
                    end
                end
            end

            dppdotg = dot( [real(dpp); imag(dpp)], [real(g); imag(g)] );
            normdpp = norm(dpp);

            if ~strncmpi(solver, 'Newton', 6)  % no conjugate is needed for newton because of using real numbers
                dpp = [dpp(1:n); conj( dpp(n+1:2*n) )];
            end

            dfzgz = D2*reshape(dpp, [], 2);
            dfgP2P = C2*reshape(dpp, [], 2);

            %%
            ls_t = 1;
            if linesearchLIM2
                maxts = arrayfun(@maxtForPhiPsy, fzgz0(:,1), fzgz0(:,2), dfzgz(:,1), dfzgz(:,2));
                ls_t = min(1, min(maxts)*0.8); % faster than min( [maxts;1] )
            end

            fQPEstim = @(t) e+ls_alpha*t*dppdotg;
            fMyFun = @(t) fMyEnergy(abs(fzgz0+t*dfzgz).^2, fgP2P+t*dfgP2P);

            if softP2P || fP2PEnergyPhiPsy(fgP2P)<1e-3 || it>min(nIter/3, 15)
                flagCanAccel = any(strfind(solver, 'AQP'));
                e_new = fMyFun(ls_t);
                while ls_t*normdpp>1e-12 && e_new > fQPEstim(ls_t)
                    ls_t = ls_t*ls_beta;
                    e_new = fMyFun(ls_t);
                end
                e = e_new;
            else
                flagCanAccel = false;
                e = fMyFun(ls_t);
            end
            
            dpp = reshape(dpp,[],2);
            
            if linesearchLIM
                ls_t = lineSearchLocallyInjectiveHarmonicMap(phipsyIters(:,1:2), dpp, fzgz0, dfzgz, ls_t, fillDistanceSegments, v, E2, L, nextSampleInSameCage);
                e = fMyFun(ls_t);
            end

            allStats(it+1, [5 7 8]) = gather( [toc*1000 fP2PEnergyPhiPsy(fgP2P+ls_t*dfgP2P)*lambda e] );

            if ls_t*normdpp<1e-12, break; end

            phi = phi + ls_t*dpp(:,1);
            psy = psy + ls_t*dpp(:,2);

            fzgz0 = fzgz0 + ls_t*dfzgz;
            fzgz2 = abs(fzgz0).^2;

            phipsyIters = [phi psy phipsyIters(:,1:end-2)];
        end

    otherwise
        warning('Unexpected solver type: %s. ', solver);
end
