fprintf('\n\n');

active_set_is_on = true; %should we use the active set approach or not
maxBisectionIterations = 10;

use_Cauchy_argument_principle = false; %using true may slows down the validation - not needed in general. use false


if hasGPUComputing
    myGPUArray = @(x) gpuArray(x);
else
    myGPUArray = @(x) x;
end
        
%preprocessing - done once
if needsPreprocessing
    v = cellfun(myGPUArray, v, 'UniformOutput', false);

    if ~exist('Phi', 'var') || numel(Phi) ~= numel(vv)
        Phi = gather(vv);
        Psy = Phi*0;
    end

    energySamples = myGPUArray(energySamples);
    denseEvaluationSamples = myGPUArray(denseEvaluationSamples);

    %% preprocess for Modulus of Continuity
	[L, indexOfMaxL] = computeLipschitzConstantOfDerivativeOfCauchy(v, denseEvaluationSamples);

    [~, SoDerivativeOfCauchyCoordinatesAtEnergySamples] = derivativesOfCauchyCoord(v, energySamples, holeCenters);

    catv = myGPUArray( cat(1, v{:}) );
    L2 = myGPUArray(zeros(numel(energySamples), numel(catv)+numel(holeCenters)));
    for i=1:numel(catv)
        L2(:, i) = distancePointToSegment(catv(i), energySamples, energySamples(nextSampleInSameCage)).^-2/2/pi;
    end

    %% Lipschitz for log basis in multiconeccted case
    for i=1:numel(holeCenters)
        L2(:, end-numel(holeCenters)+i) = distancePointToSegment(holeCenters(i), energySamples, energySamples(nextSampleInSameCage)).^-3*2;
    end

    
    %%
    DerivativeOfCauchyCoordinatesAtFixedSamples = gather(derivativesOfCauchyCoord(v, fixedSamples, holeCenters));      %copmute on gpu and transfer to cpu
    DerivativeOfCauchyCoordinatesAtEnergySamples = derivativesOfCauchyCoord(v, energySamples, holeCenters);    %copmute on gpu and transfer to cpu
    DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples = gather(derivativesOfCauchyCoord(v, activeSetPoolSamples, holeCenters)); %copmute on gpu and transfer to cpu
    DerivativeOfCauchyCoordinatesAtDenseEvaluationSamples = derivativesOfCauchyCoord(v, denseEvaluationSamples, holeCenters); %no gather - keep on the gpu

%     DerivativeOfCauchyCoordinatesAtEnergySamples = gather( DerivativeOfCauchyCoordinatesAtEnergySamples );
    
%     fillDistanceSegments = abs(circshift(denseEvaluationSamples, -1) - denseEvaluationSamples)/2; %fill distance for each segment
    fillDistanceSegments = myGPUArray( abs(energySamples-energySamples(nextSampleInSameCage))/2 );
    
    nextSampleInSameCage = myGPUArray(nextSampleInSameCage);

    %preprocess for ARAP energy
    Q = DerivativeOfCauchyCoordinatesAtEnergySamples'*DerivativeOfCauchyCoordinatesAtEnergySamples;
    %sqrtQ = sqrtm(Q);
    
    mydiag = @(x) sparse(1:numel(x), 1:numel(x), x);
    [E, vec] = eig(gather(Q));
    vec = diag(vec); vec = vec(2:end);
    ARAP_q = mydiag(vec.^0.5)*E(:,2:end)';
    clear Q;
    
    needsPreprocessing = false;

    forceConformalMode = false;
    
    phi = Phi; psy = Psy;
end

total_time = tic;


switch solver_type
    case {'CVX', 'Direct Mosek'}
        if active_set_is_on

            if(~exist('activeSetSigma1', 'var'))
                activeSetSigma1 = [];
            end
            if(~exist('activeSetSigma2', 'var'))
                activeSetSigma2 = [];
            end
            if(~exist('activeSet_k', 'var'))
                activeSet_k = [];
            end

            abs_fz = abs(DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples*Phi);
            abs_fzbar = abs(DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples*Psy);

            sigma1 = gather(abs_fz + abs_fzbar);
            sigma2 = gather(abs_fz - abs_fzbar);
            k = gather(abs_fzbar ./ abs_fz);

            upperThresholdSigma1 = 0.95*sigma1_upper_bound;
            lowerThresholdSigma1 = 0.945*sigma1_upper_bound;

            lowerThresholdSigma2 = 1.15*sigma2_lower_bound;
            upperThresholdSigma2 = 1.2*sigma2_lower_bound;

            upperThreshold_k = 0.95*k_upper_bound;
            lowerThreshold_k = 0.945*k_upper_bound;

            warning('off', 'signal:findpeaks:largeMinPeakHeight');

            [~, indicesToBeAddedSigma1] = findpeaks(sigma1, 'MinPeakHeight', upperThresholdSigma1);
            indicesToBeAddedSigma1 = union(indicesToBeAddedSigma1, find(sigma1 > sigma1_upper_bound));
            [~, indicesToBeAddedSigma2] = findpeaks(-sigma2, 'MinPeakHeight', -lowerThresholdSigma2); %we use minus to find local minimum
            indicesToBeAddedSigma2 = union(indicesToBeAddedSigma2, find(sigma2 < sigma2_lower_bound));
            [~, indicesToBeAdded_k] = findpeaks(k, 'MinPeakHeight', upperThreshold_k);
            indicesToBeAdded_k = union(indicesToBeAdded_k, find(k > k_upper_bound));


            activeSetSigma1 = union(activeSetSigma1, indicesToBeAddedSigma1);
            indicesToBeRemovedSigma1 = activeSetSigma1(sigma1(activeSetSigma1) < lowerThresholdSigma1);
            activeSetSigma1 = setdiff(activeSetSigma1, indicesToBeRemovedSigma1);

            activeSetSigma2 = union(activeSetSigma2, indicesToBeAddedSigma2);
            indicesToBeRemovedSigma2 = activeSetSigma2(sigma2(activeSetSigma2) > upperThresholdSigma2);
            activeSetSigma2 = setdiff(activeSetSigma2, indicesToBeRemovedSigma2);

            activeSet_k = union(activeSet_k, indicesToBeAdded_k);
            indicesToBeRemoved_k = activeSet_k(k(activeSet_k) < lowerThreshold_k);
            activeSet_k = setdiff(activeSet_k, indicesToBeRemoved_k);

            numActiveSetPool = size(DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples, 1);
        else
            activeSetSigma1 = [];
            activeSetSigma2 = [];
            activeSet_k = [];
        end

        DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1 = DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples(activeSetSigma1, 1:end);
        DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2 = DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples(activeSetSigma2, 1:end);
        DerivativeOfCauchyCoordinatesAtActiveSamples_k = DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples(activeSet_k, 1:end);

        frames_sigma2 = calc_frames(DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, Phi);
        frames_k = calc_frames(DerivativeOfCauchyCoordinatesAtActiveSamples_k, Phi);

        frames_fixed = calc_frames(DerivativeOfCauchyCoordinatesAtFixedSamples, Phi);
        %frames_fixed = ones(size(DerivativeOfCauchyCoordinatesAtFixedSamples, 1), 1); %reset the frames

        frames_energy = calc_frames(DerivativeOfCauchyCoordinatesAtEnergySamples, Phi);

        arap_frames_vector = gather( transpose(frames_energy)*DerivativeOfCauchyCoordinatesAtEnergySamples ); %note the use of transpose rather than '
        % g = arap_frames_vector*E(:,2:end)*mydiag(vec.^-0.5);
        ARAP_g = (arap_frames_vector*E(:,2:end)).*reshape(vec.^-0.5, 1, []);
end


optimizationTimeOnly = tic;

numVirtualVertices = size(CauchyCoordinatesAtP2Phandles, 2);
numFixedSamples = size(DerivativeOfCauchyCoordinatesAtFixedSamples, 1);
numEnergySamples = size(DerivativeOfCauchyCoordinatesAtEnergySamples, 1);
numDenseEvaluationSamples = size(DerivativeOfCauchyCoordinatesAtDenseEvaluationSamples, 1);


P2P_Deformation_Converged = 0;

switch solver_type
    case 'CVX' 
        [solverStatus, Energy_total, E_ISO, E_POSITIONAL, phi, psy] = cvx_p2p_harmonic( ...
            CauchyCoordinatesAtP2Phandles, DerivativeOfCauchyCoordinatesAtFixedSamples, ...
            DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1, DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, DerivativeOfCauchyCoordinatesAtActiveSamples_k, ...
            P2PCurrentPositions, ARAP_g, ARAP_q, frames_fixed, frames_sigma2, frames_k, ...
            p2p_weight, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, ...
            numVirtualVertices, numFixedSamples, numEnergySamples, ...
            no_output, forceConformalMode);

    case 'Direct Mosek'
        if forceConformalMode
           error('Conformal mode is currently supported only in CVX. Switch to CVX or try to reduce bound on k to a small number in order to approximate conformal');
        end

        tic
        [solverStatus, Energy_total, E_ISO, E_POSITIONAL, phi, psy] = mosek_p2p_harmonic( ...
            CauchyCoordinatesAtP2Phandles, DerivativeOfCauchyCoordinatesAtFixedSamples, ...
            DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1, DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, DerivativeOfCauchyCoordinatesAtActiveSamples_k, ...
            P2PCurrentPositions, ARAP_g, ARAP_q, frames_fixed, frames_sigma2, frames_k, ...
            p2p_weight, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, ...
            numVirtualVertices, numFixedSamples, numEnergySamples, ...
            no_output);

        
        statsAll = zeros(1, 8);
        statsAll(1, [5 7 8]) = [toc*1000 E_POSITIONAL Energy_total];

    case 'meshNewton'
        [XP2PDeform, statsAll] = meshNewton(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, numIterations, p2p_weight, energy_type, energy_parameter);
                
    case 'AQP'
        [XP2PDeform, statsAll] = meshAQP(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, numIterations);
        
    case 'SLIM'
        [XP2PDeform, statsAll] = meshSLIM(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, numIterations, p2p_weight, energy_type, energy_parameter);
        
    case 'ARAP'
        XP2PDeform = meshARAP(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, numIterations, p2p_weight);
        
    otherwise
             %{'AQP', 'SLIM', 'cuAQP single', 'cuAQP double', ...}

        if ~exist('NLO_preprocessed','var') || ~NLO_preprocessed
            D2 = myGPUArray( DerivativeOfCauchyCoordinatesAtEnergySamples );

            DtD = D2'*D2*2;         % Dirich energy is (|fz|^2 + |fzbar|^2), therefore should be a coeff 2
            Laplace = blkdiag(DtD, conj(DtD));
            P2PVtxIds_for_AQP = [];
            invM_AQP = myGPUArray(1i);
            
            %
            NLO_preprocessed = true;
            nPhiPsyIters = 2;

            phi = Phi;
            psy = Psy;
        end
        
        if isempty(phipsyIters)
            phipsyIters = repmat( myGPUArray( [Phi Psy] ), 1, nPhiPsyIters );
        end
        
        
        %% update matrix for AQP only when P2P set has changed
        if ~softP2P, p2p_weight = 0; end
        
        C2 = myGPUArray(CauchyCoordinatesAtP2Phandles);

        lastAQP_softP2P = (size(invM_AQP,1)==size(invM_AQP,2));
        if any( strfind(solver_type, 'AQP') ) && (any( setxor(P2PVtxIds, P2PVtxIds_for_AQP) ) || lastAQP_softP2P~=softP2P ) % size(invM_AQP,1)<size(D2,1)
            n = size(D2,2);
            cageSz = numel(cage);
            
            if numel(v)==1
                N = speye(2*n, 2*n-1);
            else
                isFreePhi = true(1, n);
                isFreeVar = [isFreePhi isFreePhi];
                nHoles = numel(v)-1;
                isFreeVar([n+cageSz 2*n+(-nHoles+1:0)]) = false;
                N = sparse([find(isFreeVar) 2*n+(1-nHoles:0)], [1:sum(isFreeVar) n+(1-nHoles:0)],  1, 2*n, sum(isFreeVar));
            end
            
            % old version matlab does not do sparse dense multiplication on gpuArray
            if hasGPUComputing && verLessThan('matlab', '9.2'), N = full(N); end
            
            if ~softP2P
                nP2P = size(C2,1);
                eq_lhs = [C2 conj(C2)];
                N2 = blkdiag(N,eye(nP2P));
                invM_AQP = [N zeros(size(N,1), nP2P)]*inv( N2'*[Laplace eq_lhs'; eq_lhs zeros(nP2P)]*N2 ) * N2';
            else
                CtC = [C2 conj(C2)]'*[C2 conj(C2)];
                M = Laplace+2*p2p_weight*CtC;  % p2p energy is |C*phi + conj(C*psy) - bP2P|^2, therefore coefficient 2

                invM_AQP = N*inv(N'*M*N)*N';
            end
            P2PVtxIds_for_AQP = P2PVtxIds;
        end
        
        
        if isempty(P2PCurrentPositions)
            Energy_total = 0; E_POSITIONAL = 0;
        else
        
            [phipsyIters, statsAll] = nlo_p2p_harmonic(invM_AQP, D2, C2, myGPUArray(P2PCurrentPositions), softP2P, p2p_weight, ...
                phipsyIters, energy_parameter, AQP_kappa, numIterations, solver_type, energy_type, nextSampleInSameCage, ...
                hessianSampleRate, fillDistanceSegments, v, SoDerivativeOfCauchyCoordinatesAtEnergySamples, L2);


            ppdif = norm( phipsyIters(:,1:2)-[phi psy], 'fro' )
            P2P_Deformation_Converged = gather(1*( ppdif < 1e-300 ));
            
            statsAll(:, end-1:end) = statsAll(:, end-1:end)/numEnergySamples; % energy normalization
        end
        
%         nMaxIter=5000;
%         energies = zeros(nMaxIter+1, 0);
%         solvers = {'AQP', 'Newton', 'Newton_SPDH_Slow', 'Gradient descent', 'Newton_SPDH'};
%         for is = 1:numel(solvers)
%             [phipsyConverge, eIter] = aqp_p2p_harmonic(Laplace, D2, myGPUArray(CauchyCoordinatesAtP2Phandles), myGPUArray(P2PCurrentPositions), softP2P, p2p_weight, ...
%             phipsyIters, energy_parameter, AQP_kappa, k_upper_bound, nMaxIter, solvers{is}, hessianSampleRate);
%         
%             energies(:,end+1) = eIter(end);
%             energies(1:numel(eIter),end) = eIter;
%         end
%         figuredocked; %subplot(121); subplot(122); 
%         loglog(1:nMaxIter+1, energies); hl=legend(solvers{:}); set(hl, 'Interpreter', 'none'); xlabel('#iteration'); ylabel('Energy'); title('convergence');
        
        
        phi = double(phipsyIters(:,1));
        psy = double(phipsyIters(:,2));
        solverStatus = 'solved';
end


if isempty(P2PCurrentPositions)
    Energy_total = 0; E_POSITIONAL = 0;
else
    E_POSITIONAL = statsAll(:,end-1);
    Energy_total = statsAll(:,end);
    E_ISO = Energy_total - E_POSITIONAL;
end


fprintf('Optimization time only time:%.4f\n', toc(optimizationTimeOnly));

if any(strcmpi(solver_type, {'CVX', 'Direct Mosek'}))
    validationTime = tic;
    
    LocallyInjectiveOnly = false;
    
%     [Phi, Psy, t] = validateMapBounds(L, indexOfMaxL, v, fillDistanceSegments, Phi, phi, Psy, psy, DerivativeOfCauchyCoordinatesAtDenseEvaluationSamples, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, maxBisectionIterations, unrefinedInwardCage, use_Cauchy_argument_principle);
    [Phi, Psy, t] = validateMapBoundsV2(L2, v, fillDistanceSegments, Phi, phi, Psy, psy, DerivativeOfCauchyCoordinatesAtEnergySamples, SoDerivativeOfCauchyCoordinatesAtEnergySamples, ...
                                       sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, maxBisectionIterations, unrefinedInwardCage, use_Cauchy_argument_principle, LocallyInjectiveOnly);

    fprintf('Validation time:%.4f\n', toc(validationTime));
    
    
    E_POSITIONAL = p2p_weight*norm( CauchyCoordinatesAtP2Phandles*Phi+conj(CauchyCoordinatesAtP2Phandles*Psy) - P2PCurrentPositions )^2;
    statsAll(1, 7) = gather(E_POSITIONAL);
else
    t = 1; Phi = phi; Psy = psy;
end

DeformedP2PhandlePositions = CauchyCoordinatesAtP2Phandles*Phi + conj(CauchyCoordinatesAtP2Phandles*Psy);

Phi = gather(Phi); Psy = gather(Psy);

if ~any(strcmpi(solver_type, {'AQP', 'SLIM', 'ARAP', 'meshNewton'}))
    XP2PDeform = gather(C*Phi + conj(C*Psy));
else
    solverStatus = 'solved';
end

if ~isempty(P2PCurrentPositions)
    fprintf('Solver status: %s\n', solverStatus);
    fprintf('E_ISO: %8.3e, E_POS: %8.3e, E: %8.3e\n', [E_ISO E_POSITIONAL Energy_total]');
    fprintf('Total script time:%.4f\n', toc(total_time));
    % fprintf('Fixed_samples:%d, Sigma1_samples:%d, Sigma2_samples:%d, k_samples:%d\n', numFixedSamples, size(activeSetSigma1, 1), size(activeSetSigma2, 1), size(activeSet_k, 1));
    fprintf('Vertices:%d, Energy_samples:%d, Evaluation_samples:%d\n', numVirtualVertices, numEnergySamples, numDenseEvaluationSamples);
end

%%
if update_distortions_plots
    fzgz = abs(D*[phi psy]);
    sig1 = fzgz*[1; 1];
    sig2 = fzgz*[1; -1];
    evec =  2*sum(fzgz.^2,2).*(1+fzgz.^2*[1;-1].^-2);

    if ~exist('hDistionPlot', 'var')
        figuredocked; subplot(221); h1=drawmesh(T, X); title('\sigma_1'); colorbar; subplot(222); h2=drawmesh(T, X); title('1/\sigma_2'); colorbar;
        subplot(223); h3=drawmesh(T, X); title('k'); colorbar; subplot(224); h4=drawmesh(T, X); title('E_{iso}'); colorbar; colormap(jet(8192))
        hDistionPlot = [h1 h2 h3 h4];
    end

    fUpdateCData = @(h, c) set(h,'CData',gather(c),'FaceColor', 'interp','EdgeColor','none');

    fUpdateCData( hDistionPlot(1), sig1 );
    fUpdateCData( hDistionPlot(2), 1./sig2 );
%     fUpdateCData( hDistionPlot(3), max(sig1, 1./sig2) );
    fUpdateCData( hDistionPlot(3), fzgz(:,2)./fzgz(:,1));
    fUpdateCData( hDistionPlot(4), evec );
end
