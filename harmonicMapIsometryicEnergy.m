function [e, g, h] = harmonicMapIsometryicEnergy(D, phi, psy, SPDHessian, energy_type, energy_param)

if nargin<4, SPDHessian = true; end
if nargin<5, energy_type = 'SymmDirichlet'; end
if nargin<6, energy_param = 1; end

fz2 = abs(D*phi).^2;
gz2 = abs(D*psy).^2;

s = energy_param;

switch energy_type
    case 'ARAP'
        evec = 2*(fz2 + gz2) - 4*sqrt(fz2) + 2;
    case 'BARAP'
        evec = 2*(fz2 + gz2) - 4*sqrt(fz2) + 2 + s*(fz2-gz2).^-1;
    case 'SymmDirichlet'
        evec = abs( (fz2 + gz2).*(1+(fz2 - gz2).^-2) );
        if s~=1, evec = evec.^s; end
    case 'Exp_SymmDirichlet'
        evec = exp( s*(fz2 + gz2).*(1+(fz2 - gz2).^-2) );
    case 'AMIPS'
        regularMIPS = false;
        evec = (s*2*(fz2+gz2)+1)./(fz2-gz2) + fz2-gz2;
        if ~regularMIPS, evec = exp(evec); end
%         evec = min(evec, 1e8); 
    case 'Beta'
        assert(false, 'Todo: energy/grad/hessian to be fixed');
        absfz = abs(D*phi);
        absgz = abs(D*psy);
        evec = (absfz+absgz).^2 + (absfz-absgz).^-2;
    case {'SymmARAP', 'NeoHookean'}
        assert(false, 'Todo: to be implimented');        
    otherwise
        warning('Unexpected energy type: %s. ', energy_type);
end

e = sum(evec);

if nargout>1
    fz = D*phi;
    gz = D*psy;
    n = size(D,1);

    switch energy_type
    case 'ARAP'
        alphas = [2*(1-abs(fz).^-1)     2*ones(n,1)];
    case 'BARAP'
        alphas = [2*(1-abs(fz).^-1) + s - s*(fz2-gz2).^-2       2 - s + s*(fz2-gz2).^-2];
    case 'SymmDirichlet'
        alphas = [1-((fz2-gz2).^-3).*(fz2+3*gz2)     1+((fz2-gz2).^-3).*(3*fz2+gz2)];
        if s~=1
            alphas = s*evec.^(1-1/s).*alphas; 
        end
    case 'Exp_SymmDirichlet'
%         evec = min(evec, 1e3);
        alphas = [1-((fz2-gz2).^-3).*(fz2+3*gz2)     1+((fz2-gz2).^-3).*(3*fz2+gz2)].*evec*s;
    case 'AMIPS'
        alphas = [1-(4*s*gz2+1).*(fz2-gz2).^-2    -(1-(4*s*fz2+1).*(fz2-gz2).^-2)];
        if ~regularMIPS, alphas = alphas.*evec; end
    case 'Beta'
        alphas = 1 + [absfz absgz].^-1.*[absgz-(absfz-absgz).^-3 absfz+(absfz-absgz).^-3];
    end

%     fC2Rv = @(x) [real(x); imag(x)];
%     g = 2*fC2Rv( [D'*(fz.*alphas(:,1)); D'*(gz.*alphas(:,2norm(g1-g2)))] );
    g = 2*[D'*(fz.*alphas(:,1)); conj(D'*(gz.*alphas(:,2)))];
end

if nargout>2
    %%
    m = size(D,2);
    
    switch energy_type
    case 'ARAP'
        betas = [abs(fz).^-3 zeros(n,2)];
     case 'BARAP'
        betas = [abs(fz).^-3 zeros(n,2)] + 2*s*(fz2-gz2).^-3*[1 1 -1];
    case 'SymmDirichlet'
        betas = 2*[fz2+5*gz2	5*fz2+gz2	-3*(fz2+gz2)].*(fz2-gz2).^-4;
        if s~=1, betas = alphas(:,[1 2 1]).*alphas(:, [1 2 2])*(s-1)/s./evec + betas*s.*evec.^(1-1/s); end
    case 'Exp_SymmDirichlet'
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
    
    if SPDHessian
        switch energy_type
        case {'ARAP', 'SymmDirichlet', 'Exp_SymmDirichlet'}
            i = alphas(:,1)<0;
            betas(i,1) = betas(i,1) + alphas(i,1)/2./fz2(i);
            alphas(i,1) = 0;
        otherwise
            s1s2 = (alphas+2*betas(:,1:2).*[fz2 gz2])*[1 1; 1 -1];
            lambda34 = [s1s2(:,1) sqrt(s1s2(:,2).^2+16*betas(:,3).^2.*fz2.*gz2)]*[1 1; 1 -1];
            t1t2 = (lambda34 - 2*alphas(:,1) - 4*betas(:,1).*fz2)./(4*betas(:,3).*gz2);
            eigenvec34nrm2 = fz2+gz2.*t1t2.^2;
            lambda34 = max(lambda34, 0)./eigenvec34nrm2;
            
            i = gz2>1e-50;
            alphas(i,:) = max(alphas(i,:), 0)*4;
            betas(i,:) = [sum( lambda34(i,:), 2 )-alphas(i,1)/2./fz2(i) ...
                          sum( lambda34(i,:).*t1t2(i,:).^2, 2 )-alphas(i,2)/2./gz2(i) ...
                          sum( lambda34(i,:).*t1t2(i,:), 2 )];

        end
    end
 
       
    %%
    ss1 = [real(fz).^2; real(fz).*imag(fz); real(fz).*imag(fz); imag(fz).^2].*repmat(betas(:,1),4,1) + [alphas(:,1); zeros(n*2,1); alphas(:,1)]*.5;
    ss2 = [real(gz).^2; real(gz).*imag(gz); real(gz).*imag(gz); imag(gz).^2].*repmat(betas(:,2),4,1) + [alphas(:,2); zeros(n*2,1); alphas(:,2)]*.5;
    ss3 = [real(fz).*real(gz); real(fz).*imag(gz); imag(fz).*real(gz); imag(fz).*imag(gz)].*repmat(betas(:,3),4,1);
    
    %%
    DR = [real(D) -imag(D)];
    DI = [imag(D)  real(D)];
    DRDI = [DR; DI];
    n = size(D,1);
    
    if isa(D, 'gpuArray')
        h = gpuArray.zeros(m*4);
    else
        h = zeros(m*4);
    end
    
    h(1:2*m, 1:2*m)             = DRDI'*[ DR.*ss1(1:n)+DI.*ss1(n+1:2*n); DR.*ss1(2*n+1:3*n)+DI.*ss1(3*n+1:4*n) ];
    h(end/2+1:end, end/2+1:end) = DRDI'*[ DR.*ss2(1:n)+DI.*ss2(n+1:2*n); DR.*ss2(2*n+1:3*n)+DI.*ss2(3*n+1:4*n) ];

    h(1:2*m, end/2+1:end)       = DRDI'*[ DR.*ss3(1:n)+DI.*ss3(n+1:2*n); DR.*ss3(2*n+1:3*n)+DI.*ss3(3*n+1:4*n) ];
    h(end/2+1:end, 1:2*m)       = h(1:2*m, end/2+1:end)';

    RIRI2RRII = [1:m m*2+(1:m) m+(1:m) m*3+(1:m)];
    h  = h(RIRI2RRII, RIRI2RRII)*4;
end

