function [e, g, h] = meshIsometricEnergy(D, fz, gz, Areas, energy_type, energy_param, SPDHessian, verboseMex)

% SPDHessian options: NP: vanilla hessian without projection, 
%                     KP: project K, 
%                     FP4, FP6: full projection with 4x4, 6x6 matrix
%                     suffix Matlab (e.g. KPMatlab) indicate computation in Matlab, otherwise in c++ (mex)

if nargin<5, energy_type = 'SymmDirichlet'; end
if nargin<6, energy_param = 1; end
if nargin<7, SPDHessian = 'KP'; end
if nargin<8, verboseMex = false; end

if strcmpi(energy_type, 'Exp_SymmDirichlet'), energy_type = 'ExpSD'; end

fz2 = abs(fz).^2;
gz2 = abs(gz).^2;

s = energy_param;

switch energy_type
    case 'ARAP'
        evec = 2*(fz2 + gz2) - 4*sqrt(fz2) + 2;
    case 'HOOK'
        evec =2*s*gz2./(fz2-gz2)+(1-s)*(fz2-gz2-1).^2/2;
    case 'SARAP'
        fzn = sqrt(fz2); gzn = sqrt(gz2);
        evec =(fzn+gzn-1).^2 + (1./(fzn-gzn) -1).^2;        
    case 'BARAP'
        evec = 2*(fz2 + gz2) - 4*sqrt(fz2) + 2 + s*(fz2 - gz2) + s./(fz2-gz2);
    case 'BCONF'
        evec = gz2 + s*(fz2 - gz2) + s./(fz2 - gz2);
    case 'SymmDirichlet'
        evec = abs( (fz2 + gz2).*(1+(fz2 - gz2).^-2) );
        if s~=1, evec = evec.^s; end
    case 'ExpSD'
        evec = exp( s*(fz2 + gz2).*(1+(fz2 - gz2).^-2) );
    case 'AMIPS'
        regularMIPS = false;
        evec = (s*2*(fz2+gz2)+1)./(fz2-gz2) + fz2-gz2;
        if ~regularMIPS, evec = exp(evec); end
%         evec = min(evec, 1e8); 
    case 'Beta'
        assert(false, 'Todo: energy/grad/hessian to be fixed');
        absfz = abs(fz);
        absgz = abs(gz);
        evec = (absfz+absgz).^2 + (absfz-absgz).^-2;
    otherwise
        warning('Unexpected energy type: %s. ', energy_type);
end

e = dot(evec, Areas);

if nargout>1
    n = size(D,1);

    switch energy_type
    case 'ARAP'
        alphas = [2*(1-abs(fz).^-1)     2*ones(n,1)];
    case 'HOOK'
        mu = s;
        kappa = 1-s;
        J = fz2-gz2;
        alphas = 2*mu*[-gz2 fz2]./J.^2 + kappa*(J-1)*[1 -1];
    case 'SARAP'
        sig1 = fzn+gzn;
        sig2 = fzn-gzn;  
        alphas = [((fzn.^-1).*((sig1-1) - sig2.^-3 + sig2.^-2))  ((gzn.^-1).*((sig1-1) + sig2.^-3 - sig2.^-2))];
        inz = find(gz2<1e-50);
        alphas(inz, 2) = repelem(1e16, length(inz), 1);
    case 'BARAP'
        alphas = [2*(1-abs(fz).^-1)   2*ones(n,1)] + s*(1-(fz2-gz2).^-2)*[1 -1];
    case 'BCONF'
        alphas = s*(1-(fz2-gz2).^-2)*[1 -1];
        alphas(:,2) = alphas(:,2)+1;
    case 'SymmDirichlet'
        alphas = 1+[-fz2-3*gz2     3*fz2+gz2]./(fz2-gz2).^3;
        if s~=1, alphas = s*evec.^(1-1/s).*alphas; end
    case 'ExpSD'
%         evec = min(evec, 1e3);
        alphas = (1+[-fz2-3*gz2     3*fz2+gz2]./(fz2-gz2).^3).*evec*s;
    case 'AMIPS'
        alphas = [-(4*s*gz2+1)    4*s*fz2+1]./(fz2-gz2).^2 + repmat([1 -1], n, 1);
        if ~regularMIPS, alphas = alphas.*evec; end
    case 'Beta'
        alphas = 1 + [absfz absgz].^-1.*[absgz-(absfz-absgz).^-3 absfz+(absfz-absgz).^-3];
    end

    g = D.*(fz.*alphas(:,1).*Areas) + conj(D).*(gz.*alphas(:,2).*Areas);
    g = 4*[real(g) imag(g)].';
end

if nargout>2
    switch energy_type
    case 'ARAP'
        betas = [abs(fz).^-3 zeros(n,2)];
    case 'HOOK'
        betas = mu*2*[2*gz2 2*fz2 -fz2-gz2]./J.^3 + kappa*repmat([1 1 -1], n, 1);
    case 'SARAP'
        betas = [0.5*(fzn.^-3).*(sig2.^-3 - sig2.^-2 -sig1 +1)+0.5*(fz2.^-1).*((sig2.^-4)*3- 2*sig2.^-3 +1) ...
                 0.5*(gzn.^-3).*(-sig2.^-3 + sig2.^-2 -sig1 +1)+0.5*(gz2.^-1).*((sig2.^-4)*3 -2*sig2.^-3 +1) ...
                (2*fzn.*gzn).^-1.*(1-3*sig2.^-4 + 2*sig2.^-3)];
        inz = find(gz2<1e-50);
        betas(inz,:) = repelem([2 10 -6], nnz(inz), 1);
    case 'BARAP'
        betas = [abs(fz).^-3 zeros(n,2)] + 2*s*(fz2-gz2).^-3*[1 1 -1];
    case 'BCONF'
        betas = [1 1 -1]*2*s.*(fz2-gz2).^-3;
    case 'SymmDirichlet'
        betas = 2*[fz2+5*gz2	5*fz2+gz2	-3*(fz2+gz2)].*(fz2-gz2).^-4;
        if s~=1, betas = alphas(:,[1 2 1]).*alphas(:, [1 2 2])*(s-1)/s./evec + betas*s.*evec.^(1-1/s); end
    case 'ExpSD'
         betas = 2*[fz2+5*gz2	5*fz2+gz2	-3*(fz2+gz2)].*(fz2-gz2).^-4;
         betas = alphas(:,[1 2 1]).*alphas(:,[1 2 2])./evec + betas.*evec*s;
    case 'AMIPS'
        betas = 2*[4*s*gz2+1  4*s*fz2+1].*(fz2-gz2).^-3;
        betas(:,3) = -mean(betas,2);
        if ~regularMIPS, betas = alphas(:,[1 2 1]).*alphas(:,[1 2 2])./evec + betas.*evec;  end
    case 'Beta'
        betas = .5* [absfz.^-3.*((absfz-absgz).^-4.*(4*absfz-absgz) - absgz) ...
                     absgz.^-3.*((absfz-absgz).^-4.*(4*absgz-absfz) - absfz) ...
                    (absfz.*absgz).^-1.*(1-3*(absfz-absgz).^-4)];
    end
    
    
    tt = tic;
    if strncmpi(SPDHessian, 'KP', 2)  % KP, or KPMatlab
        if any( strcmpi( {'ARAP','SymmDirichlet', 'ExpSD'}, energy_type ) & [1 s>=1 1] )
            %% simple modification
            i = alphas(:,1)<0;
            betas(i,1) = betas(i,1) + alphas(i,1)/2./fz2(i);
            alphas(i,1) = 0;
        else
            %% general modification
            s1s2 = (alphas+2*betas(:,1:2).*[fz2 gz2])*[1 1; 1 -1];
            lambda34 = [s1s2(:,1) sqrt(s1s2(:,2).^2+16*betas(:,3).^2.*fz2.*gz2)]*[1 1; 1 -1];
            t1t2 = (lambda34 - 2*alphas(:,1) - 4*betas(:,1).*fz2)./(4*betas(:,3).*gz2);
            evec34nrm2 = fz2+gz2.*t1t2.^2;
            lambda34 = max(lambda34, 0)./evec34nrm2;
            
            i = gz2>1e-50;

            alphas(i,:) = max(alphas(i,:), 0);
            betas(i,:) = 1/4*[sum( lambda34(i,:), 2 )-alphas(i,1)*2./fz2(i) ...
                              sum( lambda34(i,:).*t1t2(i,:).^2, 2 )-alphas(i,2)*2./gz2(i) ...
                              sum( lambda34(i,:).*t1t2(i,:), 2 )];
        end
    elseif strncmpi(SPDHessian, 'CM', 2)  % CM or CMMatlab
        switch energy_type
        case 'HOOK'
            mu = s; kappa = 1-s;
            hv = -mu*(fz2+gz2).*(fz2-gz2).^-2 + kappa*(fz2-gz2-1);
            alphas = mu./(fz2-gz2) + [max(hv,0)  -min(hv,0)];
        case 'SARAP'
            gv = fzn-gzn;
            hvvm = 6*gv.^-4 - 4*gv.^-3; 
            hvvm(gv<=1.5) = 0;
            i1 = alphas(:,1)<0; 
            i2 = alphas(:,2)<0;
            betas(i1, 1) = betas(i1, 1)+alphas(i1,1)./(2*fz2(i1));
            betas(i2, 2) = betas(i2, 2)+alphas(i2,2)./(2*gz2(i2));
            alphas(i1, 1) = 0;
            alphas(i2, 2) = 0;
            bnz = gzn>1e-40;
            betas(bnz,:) = betas(bnz,:) - hvvm(bnz)/4.*[fz2(bnz) gz2(bnz) -fzn(bnz).*gzn(bnz)];
        case {'SymmDirichlet', 'ExpSD', 'ARAP'}
            if ~strcmpi(energy_type, 'SymmDirichlet') || s>=1
                i1 = alphas(:,1)<0;
                betas(i1,1) = betas(i1,1)+alphas(i1,1)./(2*fz2(i1));
                alphas(i1,1) =  0;
            else
                assert(false, 'power SD energy (en^s) with s<1 is not supported by CM'); 
            end
        otherwise 
            assert(false, 'Energy %s is not implemented for CM', energy_type);
        end
    end
 
    switch lower(SPDHessian)  % case insenstive
        case {'kp', 'np', 'cm'} % mex
            h = projMeshHessians((alphas.*Areas*2)', (betas.*Areas*2)', fz, gz, D.', 0, verboseMex);
        case 'fp4'
            h = projMeshHessians((alphas.*Areas*2)', (betas.*Areas*2)', fz, gz, D.', 1, verboseMex);
        case 'fp6'
            h = projMeshHessians((alphas.*Areas*2)', (betas.*Areas*2)', fz, gz, D.', 2, verboseMex);
        case {'kpmatlab', 'npmatlab'}  % assemble local hessians in matlab
            ss1 = [real(fz).^2 real(fz).*imag(fz) real(fz).*imag(fz) imag(fz).^2].*repmat(betas(:,1),1,4) + [alphas(:,1) zeros(n,2) alphas(:,1)]*.5;
            ss2 = [real(gz).^2 real(gz).*imag(gz) real(gz).*imag(gz) imag(gz).^2].*repmat(betas(:,2),1,4) + [alphas(:,2) zeros(n,2) alphas(:,2)]*.5;
            ss3 = [real(fz).*real(gz) real(fz).*imag(gz) imag(fz).*real(gz) imag(fz).*imag(gz)].*repmat(betas(:,3),1,4);

            %%
            A = [ss1(:,1:2) ss3(:,1:2) ss1(:,4) ss3(:,3:4) ss2(:,1:2) ss2(:,4)];
            B = [A(:,[1 3 8]) *[1; 2;1]   A(:,[5 7 10])*[1;-2;1]  A(:,[2 4 6 9])*[-1;1;-1;1] ...
                 A(:,[5 7 10])*[1; 2;1]   A(:,[1 3 8 ])*[1;-2;1]  A(:,[2 4 6 9])*[1;1;-1;-1] ...
                 A(:,[2 4 6 9])*[1 -1; 1 1; 1 1; 1 -1]   A(:,[1 8])*[1;-1]  A(:,[5 10])*[-1;1]];

            fMyProd = @(x, y) [x(:,1).*y x(:,2).*y x(:,3).*y];
            RR = fMyProd(real(D), real(D));
            II = fMyProd(imag(D), imag(D));
            RI = fMyProd(real(D), imag(D));
            IR = fMyProd(imag(D), real(D));
            
            %%
            B = B.*Areas*8;
            h = [ B(:,1).*RR + B(:,2).*II + B(:,3).*(RI + IR) ...
                  B(:,4).*RR + B(:,5).*II + B(:,6).*(RI + IR) ...
                  B(:,7).*RR + B(:,8).*II + B(:,9).*RI + B(:,10).*IR ...
                  B(:,7).*RR + B(:,8).*II + B(:,10).*RI + B(:,9).*IR ];

            % reorder based on full 6x6 hessian, swithch to column based storage, to avoid transpose after mex code
            h = h(:, [1:3 19:21 4:6 22:24 7:9 25:30 10:12 31:33 13:15 34:36 16:18])';
        case {'fp4matlab', 'fp6matlab'}  % full projection in Matlab
            fMakeK = @(f, g, a, b) [4*f*f'*b(1)+2*a(1)*eye(2) 4*f*g'*b(3); 4*g*f'*b(3) 4*g*g'*b(2)+2*a(2)*eye(2)];
            h = zeros(36, n);
            for i = 1:n
               Dr = real(D(i,:)); Di = imag(D(i,:));
               M = [Dr Di;  -Di Dr; Dr -Di;  Di Dr];
               K = fMakeK([real(fz(i)); imag(fz(i))], [real(gz(i)); imag(gz(i))], alphas(i,:), betas(i,:));

               if strncmpi(SPDHessian, 'fp4', 3)
                   vv = [norm(Dr)^2-norm(Di)^2 norm(Dr)^2+norm(Di)^2 2*Dr*Di'];
                   a = vv(1)/vv(2);
                   b = vv(3)/vv(2);
                   c = sqrt(1-a^2-b^2);
                   R = [eye(2) [a b; -b a]; zeros(2) c*eye(2)];
                   iR = [eye(2) [a b; -b a]/-c; zeros(2) 1/c*eye(2)];

                   K = R*K*R';
                   [Ke, Kv] = eig( tril(K) + tril(K,-1)' );
                   K = iR*Ke*max(Kv,0)*Ke'*iR';
               end
               
               H = M'*K*M*Areas(i)*2;
                   
               if strncmpi(SPDHessian, 'fp6', 3)
                   [He, Hv] = eig( tril(H) + tril(H,-1)' );
                   H = He*max(Hv,0)*He';
               end
               
               h(:,i) = reshape(H,[],1);
            end
        otherwise
            warning('Unexpected Hessian projection: %s. ', SPDHessian);
    end
    
    fprintf('%-30s %fs\n', SPDHessian, toc(tt));
end

