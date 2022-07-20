function [phipsy] = MetricInterp_Multi(v,C2,D2,phipsy_0,MG_k,fillDistanceSegments,E2,L2,nextSampleInSameCage,numHessianSamplesRate,isGPUcompute)

niter = 100;

% isGPUcompute = true;

if isGPUcompute
     vv = cat(1, v{:});
     cageSizes = gather( int32( cellfun(@numel, v) ) );
     cageOffsets = cumsum([0 cageSizes(1) cageSizes(2:end)-2]);
     hessian_sample_indices = floor((0:floor(size(D2,1)*numHessianSamplesRate-1))/numHessianSamplesRate);
     params = struct('sample_spacings_half',gpuArray(fillDistanceSegments),'v',gpuArray(vv),'E2',gpuArray(E2),'L',gpuArray(L2),'nextSampleInSameCage',gpuArray(int32(nextSampleInSameCage-1)),'hessian_sample_indices',gpuArray(int32(hessian_sample_indices)),'nIter',niter);
     
     tic
     [phipsy_k,tt_itr] = cuHarmonicInterp(gpuArray(complex(D2)),gpuArray(complex(phipsy_0)),gpuArray(MG_k),params,cageOffsets);
     toc;
     
     phipsy = phipsy_k;
     
     fprintf("num of itr = %d\n",tt_itr);
else

    hessian_sample_indices = floor((0:floor(size(D2,1)*numHessianSamplesRate)-1)/numHessianSamplesRate) + 1;

    % nullspace, reduce problem size by removing redundant vars
    n = size(D2,2);
    if ~iscell(v)
        N = speye(2*n, 2*n-1);

        % for real variables
        Nr = blkdiag(N, N);
        Nr(end-numel(v)+2:end,:) = -Nr(end-numel(v)+2:end,:);
        Nr(:,[1 size(N,2)+1]) = [];
    else
        cageSz = numel(v{1});
        isFreeVar = true(1, n*2);
        nHoles = numel(v)-1;

        %decomposition & additional term & rotation
        isFreeVar([n+cageSz 2*n+(1-nHoles:0)]) = false;
        N = sparse([find(isFreeVar) 2*n+(1-nHoles:0)], [1:sum(isFreeVar) n+(1-nHoles:0)],  1, 2*n, sum(isFreeVar));
        Nr = blkdiag(N, N);
        Nr(end-numel(v)+2:end,:) = -Nr(end-numel(v)+2:end,:);
        Nr(:,[1 size(N,2)+1]) = [];
    end

    fC2Rm = @(x) [real(x) -imag(x); imag(x) real(x)];
    fR2Cv = @(x) complex(x(1:end/2),x(end/2+1:end));
    fC2Rv = @(x) [real(x);imag(x)];

    phi_k = phipsy_0(:,1);
    psy_k = phipsy_0(:,2);
    phipsy_k = [phi_k psy_k];

    fzgz = D2*phipsy_k;

    g1 = MG_k(:,1);g2 = MG_k(:,2);g3 = MG_k(:,3);

    tic
    for k = 1:niter

        [e,h,g] = harmonicMapIsometryicEnergyInterp3(D2,phi_k,psy_k,MG_k);

        g = fR2Cv(g);

        try
            dpp = fR2Cv( Nr*( (Nr'*h*Nr)\(Nr'*fC2Rv(-g)) ) );
        catch
            h = h + 0.001*eye(4*n);
            dpp = fR2Cv( Nr*( (Nr'*h*Nr)\(Nr'*fC2Rv(-g)) ) );
            error('singular H\n');
        end

    %     dpp = fR2Cv( Nr*( (Nr'*h*Nr)\(Nr'*fC2Rv(-g)) ) );
        dppdotg = dot( [real(dpp); imag(dpp)], [real(g); imag(g)] );
        normdpp = norm(dpp); 
        dpp = reshape(dpp,[],2);
        dfzgz = D2*dpp;

        maxts = arrayfun(@maxtForPhiPsy, fzgz(:,1), fzgz(:,2), dfzgz(:,1), dfzgz(:,2));
        ls_t = min(1, min(maxts)*0.8);

        fQPEstim = @(t) e+0.2*t*dppdotg;
        fMyEnergy = @(f1,f2,f3)  sum(( g3.*f1 - 2*g2.*f2 + g1.*f3 ).*( 1./(g1.*g3 - g2.^2) + 1./(f1.*f3 - f2.^2) )); 
        f1 = @(fz,gz) (real(fz) + real(gz)).^2 + (imag(fz) - imag(gz)).^2;
        f2 = @(fz,gz) -2*(real(fz).*imag(gz) + real(gz).*imag(fz));
        f3 = @(fz,gz) (real(fz) - real(gz)).^2 + (imag(fz) + imag(gz)).^2;
        fMyFun = @(t) fMyEnergy(f1(fzgz(:,1)+t*dfzgz(:,1),fzgz(:,2)+t*dfzgz(:,2)),f2(fzgz(:,1)+t*dfzgz(:,1),fzgz(:,2)+t*dfzgz(:,2)),f3(fzgz(:,1)+t*dfzgz(:,1),fzgz(:,2)+t*dfzgz(:,2)));

        e_new = fMyFun(ls_t);
        while ls_t*normdpp>1e-12 && e_new > fQPEstim(ls_t)
            ls_t = ls_t*0.5;
            e_new = fMyFun(ls_t);
        end
        e = e_new;

        ls_t = lineSearchLocallyInjectiveHarmonicMap(phipsy_k, dpp, fzgz, dfzgz, ls_t, fillDistanceSegments, v, E2, L2, nextSampleInSameCage);

        fprintf("ls_t*normdpp = %d\n",ls_t*normdpp);
        if ls_t*normdpp<1e-6, break;end

        phi_k = phi_k + ls_t*dpp(:,1);
        psy_k = psy_k + ls_t*dpp(:,2);
        phipsy_k = [phi_k psy_k];

        fzgz = fzgz + ls_t*dfzgz;
    end

    phipsy = phipsy_k;

    fprintf("num of itr = %d\n",k);
    
    toc;
end

end

