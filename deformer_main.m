fC2R = @(x) [real(x) imag(x)];
fR2C = @(x) complex(x(:,1), x(:,2));

%%
if ~exist('working_dataset', 'var') || isempty(working_dataset)
    working_dataset = 'annulus';
    fprintf('working shape default: %s\n', working_dataset);
end

datadir = fullfile(cd, 'data', working_dataset, '\');

if exist('numMeshVertex', 'var')~=1 
    numMeshVertex = 10000;
    fprintf('numMeshVertex default: %d\n', numMeshVertex);
end

datafile = fullfile(datadir, 'data.mat');
imgfilepath  = fullfile(datadir, 'image.png');

P2Psrc = zeros(0,1); P2Pdst = zeros(0,1);

if exist(datafile, 'file') == 2
    %% load presaved data
    load(datafile);
else    
    % read image dimension
    iminfo = imfinfo(imgfilepath);
    img_w = iminfo.Width;
    img_h = iminfo.Height;
    
    %% use predefined cage
    cagefilepath = fullfile(datadir, 'cage.obj');
    if exist(cagefilepath, 'file') == 2
        [cx, cf] = readObj( cagefilepath, true );
        allcages = cellfun( @(f) fR2C(cx(f,:)), cf, 'UniformOutput', false );
    else
        %% extract cage from image
        offset = 10;
        simplify = 10;
        allcages = GetCompCage(imgfilepath, offset, simplify, 0, 0);

        allcages = allcages( cellfun(@(c) numel(c)>3&&isempty( selfintersect(real(c), imag(c)) ), allcages) );
        
        for i=1:numel( allcages )
            curcage = allcages{i};
            % remove tail if repeating head
            if abs(curcage(1)-curcage(end))/sum( abs(curcage-curcage([2:end 1])) )<1e-4, curcage = curcage(1:end-1); end

            % make sure cage and holes have correct orientation
            if xor(i==1, signedpolyarea(curcage)>0); curcage = curcage(end:-1:1); end

            % make sure first two entries in the holes are sufficiently different, hole(2) - hole(1) \ne 0, 
            if i>1
                [~, imax] = max( abs(curcage-curcage([2:end 1])) );
                curcage = curcage([imax:end 1:imax-1]);
            end

            allcages{i} = curcage;
        end

        allcages = allcages(cellfun(@(x) numel(x)>2, allcages));

    end
end

cage = allcages{1};
holes = reshape( allcages(2:end), 1, [] );

[X, T] = cdt([cage, holes], [], numMeshVertex, false);
X = fR2C(X);

%% load p2p
P2PVtxIds = triangulation(T, fC2R(X)).nearestNeighbor( fC2R(P2Psrc) );
P2PCurrentPositions = P2Pdst;
iP2P = 1;

%% texture
uv = fR2C([real(X)/img_w imag(X)/img_h])*100 + complex(0.5, 0.5);

%%
% fDrawPoly = @(x, c) plot(real(x([1:end 1])), imag(x([1:end 1])), c);
% figuredocked;  fDrawPoly(v, 'r-');
% hold on;  fDrawPoly(offsetcage, 'b-');
% hm = drawmesh(t,x);

%% for Key Frame interpolation
if exist([datadir 'PhiPsyKF.mat'], 'file') == 2
    load([datadir 'PhiPsyKF']);
    fprintf('%d frames are loaded\n', size(PhiPsyKF,2)/2);

    if exist('anchorsForInterp', 'var') == 1 && ~isempty(anchorsForInterp)
        interpAnchID = triangulation(T, fC2R(X)).nearestNeighbor( fC2R(anchorsForInterp) );
    end
end

ikeyframe = 2;
BDHIAlpha = 1;

%% deformation based on symmetrized Dirichlet energy minimization
if ~hasGPUComputing, warning('NO CUDA capable GPU is present, the computation will fall back to CPU!'); end

harmonic_map_solvers = {'bemAQP', 'bemSLIM', 'AQP', 'SLIM', 'meshNewton', 'Newton', 'Newton_SPDH', 'Newton_ConformalMap', 'Newton_SPDH_FullEig', 'Gradient Descent', 'LBFGS', 'CVX', 'Direct Mosek', ...
                        'cuAQP', 'cuGD', 'cuNewton', 'cuNewton_SPDH', 'cuNewton_SPDH_FullEig'...
                        %'cuAQP single', 'cuGD single', 'cuNewton single', 'cuNewton_SPDH single'
                        };

if hasGPUComputing                    
    default_harmonic_map_solver = 'cuNewton_SPDH';
else
    warning('no cuda capable GPU present, switching to CPU solver');
    default_harmonic_map_solver = 'Newton_SPDH';
end

harmonic_map_energies = {'ARAP', 'BARAP', 'SymmDirichlet', 'Exp_SymmDirichlet', 'AMIPS', 'Beta', 'SymmARAP', 'NeoHookean'};
default_harmonic_map_energy = 'SymmDirichlet';

update_distortions_plots = false;
