% interpolation results for multi-connected domain
%%
assert( all(signedAreas(X, T)>0), 'source triangulation not in correct order!');

%% PreCompute
if exist('needPreCompute', 'var')~=1, needPreCompute = false; end

if needPreCompute
    nBEMSample = 20;
    BEMSampleOffset = cage_offset; 

    if exist('nVirtual', 'var')~=1, nVirtual = 1; end

    nVirtual = 1;
    mxsub = @(x) subdivPolyMat(x, nVirtual*size(x,1));

    numRealVirtualVertex = size(mxsub, 1);

    nx = @(x) size(x,1);
    S = @(x) subdivPolyMat(mxsub(x)*x,ceil(nBEMSample*nx(x)));
    w = @(x) S(x)*mxsub(x)*polygonOffset(x, BEMSampleOffset, false);
    W = reshape(cellfun(w,v,'UniformOutput',false),[],1);
    W = cell2mat(W);

    %% init, precomputations 
    C = cauchyCoordinates(v, X,holeCenters);
    D = derivativesOfCauchyCoord(v, X,holeCenters);

    %% compute cauchy coordinates and its derivatives for samples
    fSampleOnPolygon = @(n, p) subdivPolyMat(p, n)*p;

    fPerimeter = @(x) sum( abs(x-x([2:end 1])) );
    sumCagePerimeter = sum( cellfun(fPerimeter, v) );

    energySamples = cellfun(@(h) fSampleOnPolygon(ceil(fPerimeter(h)/sumCagePerimeter*numEnergySamples), h), [cage holes], 'UniformOutput', false);
    nSamplePerCage = cellfun(@numel, energySamples);
    energySamples = cat(1, energySamples{:});

    nextSampleInSameCage = [2:sum(nSamplePerCage) 1];   % next sample on the same cage, for Lipschitz constants and correct sample spacing computation
    nextSampleInSameCage( cumsum(nSamplePerCage) ) = 1+cumsum( [0 nSamplePerCage(1:end-1)] );

    C2 = cauchyCoordinates(v, energySamples, holeCenters);
    [D2, E2] = derivativesOfCauchyCoord(v, energySamples, holeCenters);
    fillDistanceSegments = abs(energySamples-energySamples(nextSampleInSameCage))/2;

    catv = myGPUArray( cat(1, v{:}) );
    L2 = myGPUArray(zeros(numel(energySamples), numel(catv)+numel(holeCenters)));
    for i=1:numel(catv)
        L2(:, i) = distancePointToSegment(catv(i), energySamples, energySamples(nextSampleInSameCage)).^-2/2/pi;
    end

    %% Lipschitz for log basis in multiconeccted case
    for i=1:numel(holeCenters)
        L2(:, end-numel(holeCenters)+i) = distancePointToSegment(holeCenters(i), energySamples, energySamples(nextSampleInSameCage)).^-3*2;
    end
end

C0 = C2;
D0 = D2;
E2 = SoDerivativeOfCauchyCoordinatesAtEnergySamples;
% precomputed data
invC0 = pinv( [real(C0) -imag(C0)] );
invC0 = complex(invC0(1:size(invC0,1)/2,:), invC0(size(invC0,1)/2+1:size(invC0,1),:));
invD0 = pinv(D0);

Phi = phi;Psy = psy;
if ~exist('PhiPsyKF', 'var')
    PhiPsyKF = [vv*[1 0] Phi Psy];
end

phipsy = PhiPsyKF;
numKeyFrame = size(phipsy,2)/2;
ikeyframe = min(numKeyFrame-1, ikeyframe);

keyframes = (1:numKeyFrame)*2-1;

if ~var(keyframes), warning('interpolating same shape!'); end

fprintf('### interplating %d keyframes\n', numKeyFrame);

if 1
    fz = D0*phipsy(:, keyframes);
    fzbar = D0*phipsy(:, keyframes+1);  % fzbarbar actaully
    
    fzX = D*phipsy(:, keyframes);
    fzbarX = D*phipsy(:, keyframes+1);
    
end

fWtFun = @(w) linearWeight(numKeyFrame, w);

fComposeHarmonicMap = @(phipsy) phipsy(:,1:2:end)+conj(phipsy(:,2:2:end));
%% extract g
gFromFz = @(fz) complex( log(abs(fz)), angle(fz(end)) + cumsum(angle(fz./circshift(fz, 1))) );
g = gFromFz(fz);


TR = triangulation(T, real(X), imag(X));
edges = TR.edges;
nv = numel(X);

%% for numerical integration
if exist('interpAnchID', 'var') && ~isempty(interpAnchID)
    anchorVertexIndex = interpAnchID(1);   
else
    anchorVertexIndex = 1; 
end
    
adjacencyGraph = sparse(edges(:, 2), edges(:, 1), 1, nv, nv);
assert(nnz(triu(adjacencyGraph)) == 0);
[disc, pred, closed] = graphtraverse(adjacencyGraph, anchorVertexIndex, 'Directed', false, 'Method', 'BFS');
endIndices = uint32(disc(2:end))';
startIndices = uint32(pred(endIndices))';
eVec = X(endIndices) - X(startIndices);
e4treecumsum = [startIndices endIndices];

gX0 = LogFzFromTreeRoute(fzX(:,1),e4treecumsum,uint32(anchorVertexIndex));
gX1 = LogFzFromTreeRoute(fzX(:,2),e4treecumsum,uint32(anchorVertexIndex));
gX = [gX0 gX1];


% bdhiMethod = ('nu', 'eta', 'metric');
if exist('bdhiMethod', 'var')~=1, bdhiMethod = 'metric'; end

fprintf('### prepare to do BDH interpolate by %s\n', bdhiMethod);

if strcmp( bdhiMethod, 'nu' )
    %% interp nu, does not maintain/interp stretch direction
    fInterpFz = @(wt) exp(g*fWtFun(wt));
    nu = fzbar./fz;
    fInterpNu = @(wt) nu*fWtFun(wt);
    fInterpFzbar = @(wt) fInterpFz(wt).*fInterpNu(wt);
    
    nuX = fzbarX./fzX;
    fInterpNuX = @(wt) nuX*fWtFun(wt);
    fInterpFzX = @(wt) exp(gX*fWtFun(wt));
    fInterpFzbarX = @(wt) fInterpFzX(wt).*fInterpNuX(wt);
    fInterpEtaX = @(wt) fInterpFzX(wt).^2.*fInterpNuX(wt);
    
    fInterpEta = @(wt) fInterpFz(wt).*fInterpFzbar(wt);
    fA = @(wt) fInterpEta(wt);
    fB = @(wt) abs(fInterpFz(wt)).^2 + abs(fInterpFzbar(wt)).^2;
    
    fInterpM = @(wt) [fB(wt)+2*real(fA(wt)) -2*imag(fA(wt)) fB(wt)-2*real(fA(wt))];
elseif strcmp( bdhiMethod, 'eta' )
    scaleEta = 1;
    scaleGlobal = 1;
    boundGlobalk = 0;

    if boundGlobalk
        kmax = max(abs(fzbar(:)./fz(:)));
    else
        kmax = max(abs(fzbar./fz), [], 2);
    end

    %% interp with BEM
    fInterpEta0  = @(wt) (fz.*fzbar)*fWtFun(wt);
    fInterpFz0   = @(wt) exp(g*fWtFun(wt));
    fInterpk0    = @(wt) abs(fInterpEta0(wt)./fInterpFz0(wt).^2);
    fInterpFzbar0 = @(wt) fInterpEta0(wt)./fInterpFz0(wt);

    fMyHilbert = @(s) C0*invC0*s - 1i*mean(imag(C0*invC0*s));
    if scaleGlobal
        maxsiga = max( abs(fz)+abs(fzbar), [], 2 );
        minsigb = min( abs(fz)-abs(fzbar), [], 2 );

        fRhoa = @(wt) (maxsiga-abs(fInterpFz0(wt)))./abs(fInterpFzbar0(wt));
        fRhob = @(wt) (abs(fInterpFz0(wt))-minsigb)./abs(fInterpFzbar0(wt));

        fSScale = @(wt) min(1, min( min([kmax./fInterpk0(wt), fRhoa(wt), fRhob(wt)], [], 2) ) );
        fSScale = @(wt) 1; % for bounded distortion 
    else
        fSScale = @(wt) exp( fMyHilbert( min(0, log(kmax./fInterpk0(wt))) ) );
    end

    fInterpEta   = fInterpEta0;
    fInterpFz    = fInterpFz0;

    fInterpEtaX0  = @(wt) fzX.*fzbarX*fWtFun(wt);
    fInterpFzX0   = @(wt) exp(gX*fWtFun(wt));
    fInterpEtaX  = fInterpEtaX0;
    fInterpFzX   = fInterpFzX0;
    
    if ~scaleEta
        fInterpFz   = @(wt) fSScale(wt).^-0.5.*fInterpFz0(wt);
        fInterpFzX  = @(wt) fSScale(wt).^-0.5.*fInterpFzX0(wt);
    else
        fInterpEta  = @(wt) fSScale(wt).*fInterpEta0(wt);
        fInterpEtaX = @(wt) fSScale(wt).*fInterpEtaX0(wt);
    end

    fInterpFzbar = @(wt) fInterpEta(wt)./fInterpFz(wt);
    
    fA = @(wt) fInterpEta(wt);
    fB = @(wt) abs(fInterpFz(wt)).^2 + abs(fInterpFzbar(wt)).^2;
    
    fInterpM = @(wt) [fB(wt)+2*real(fA(wt)) -2*imag(fA(wt)) fB(wt)-2*real(fA(wt))];
else
     %% abs2(fz) interp   
     eta = fz.*fzbar;
     dfnorm2 = abs(fz).^2 + abs(fzbar).^2;
    
     fA = @(wt) eta*fWtFun(wt);
     fB = @(wt) dfnorm2*fWtFun(wt);
    
     fInterpM = @(wt) [fB(wt)+2*real(fA(wt)) -2*imag(fA(wt)) fB(wt)-2*real(fA(wt))];
end

if ~exist('interpAnchID', 'var'), interpAnchID = []; end
if isempty(interpAnchID), warning('anchors for interpolation not set!'); end
% if isempty(interpAnchID),interpAnchID = [1 100];end
% anchX = fComposeHarmonicMap( C(interpAnchID, :)*phipsy(:, keyframes(1)+(0:1)) );
anchSrc = fComposeHarmonicMap( C(interpAnchID, :)*phipsy(:, reshape([keyframes; keyframes+1], [], 1)) );

fInterpPhiPsy = @(wt)MetricInterp_Multi(v,C,D2,PhiPsyKF(:,1:2),fInterpM(wt),fillDistanceSegments,E2,L2,nextSampleInSameCage,bdhmiHessianSampleRate,bdhmiUseGPU);

fInterpPhiPsy3 = @(PhiPsyIterMulti,wt)MetricInterp_Multi(v,C,D2,PhiPsyIterMulti,fInterpM(wt),fillDistanceSegments,E2,L2,nextSampleInSameCage,numHessibdhmiHessianSampleRateanSamplesRate,bdhmiUseGPU);

fBdhInterpX2 = @(wt) fComposeHarmonicMap(C*fInterpPhiPsy(wt));

fBdhInterpX = @(wt) alignPoseByAnchors( fBdhInterpX2(wt), interpAnchID, anchSrc, fWtFun(wt) );

%% comparison SIG13/ARAP/FFMP interp
% X01 = { fC2R( fComposeHarmonicMap(C*phipsy(:,keyframes(1)+(0:1))) ) };

X01 = arrayfun(@(i) fC2R( fComposeHarmonicMap( C*phipsy(:,i:i+1) ) ), keyframes, 'UniformOutput', false);

if numel(interpAnchID)>1
    X01 = cellfun(@(x) x+repmat(X01{1}(interpAnchID(1),:)-x(interpAnchID(1),:), size(x,1), 1), X01, 'UniformOutput', false);
end


fSIG13Interp= @(wt) fR2C( metricInterp(X01, T, fWtFun(wt)', struct('anchors', interpAnchID, 'metric', 'metric tensor')) );
fARAPInterp = @(wt) fR2C( arapInterp(X01, T, fWtFun(wt)', struct('anchors', interpAnchID)) );
fARAPLGInterp = @(wt) fR2C( arapLGInterp(X01, T, fWtFun(wt)', struct('anchors', interpAnchID)) );
fFFMPInterp = @(wt) fR2C( FFMPInterp(X01, T, fWtFun(wt)', struct('anchors', interpAnchID)) );
fGBDHInterp = @(wt) fR2C( GBDHInterp(X01, T, fWtFun(wt)', struct('anchors', interpAnchID)) );
fANALYTICInterp = @(wt)  alignPoseByAnchors( fR2C( analyticInterp(X01, T, fWtFun(wt)',X01{1}) ), interpAnchID, anchSrc, fWtFun(wt) );