function [e,h,g] = harmonicMapIsometryicEnergyInterp(D, phi, psy, MG)

g1 = MG(:,1);g2 = MG(:,2);g3 = MG(:,3);

fz = complex(D*phi);
gz = complex(D*psy);

f1 = (real(fz) + real(gz)).^2 + (imag(fz) - imag(gz)).^2;
f2 = -2*(real(fz).*imag(gz) + real(gz).*imag(fz));
f3 = (real(fz) - real(gz)).^2 + (imag(fz) + imag(gz)).^2;

evec = ( g3.*f1 - 2*g2.*f2 + g1.*f3 ).*( 1./(g1.*g3 - g2.^2) + 1./(f1.*f3 - f2.^2) );

e = sum(evec);

n = size(D,2);
m = size(D,1);
% mh = length(hessian_sample_indices);

Alpha1 = g3.*( 1./(f1.*f3 - f2.^2) + 1./(g1.*g3 - g2.^2) ) - ( f3.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^2;
Alpha2 = -2*g2.*( 1./(f1.*f3 - f2.^2) + 1./(g1.*g3 - g2.^2) ) + ( 2*f2.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^2;
Alpha3 = g1.*( 1./(f1.*f3 - f2.^2) + 1./(g1.*g3 - g2.^2) ) - ( f1.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^2;

Beta11 = -(2*f3.*g3)./(f1.*f3 - f2.^2).^2 + ( 2*f3.^2.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^3;
Beta12 = (2*f2.*g3)./(f1.*f3 - f2.^2).^2 + (2*f3.*g2)./(f1.*f3 - f2.^2).^2 - ( 4*f2.*f3.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^3;
Beta13 = -(f1.*g3 - 2*f2.*g2 + f3.*g1)./(f1.*f3 - f2.^2).^2 - (f1.*g3)./(f1.*f3 - f2.^2).^2 - (f3.*g1)./(f1.*f3 - f2.^2).^2 + ( 2*f1.*f3.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^3;
Beta21 = Beta12;
Beta22 = 2*(f1.*g3 - 2*f2.*g2 + f3.*g1)./(f1.*f3 - f2.^2).^2 - (8*f2.*g2)./(f1.*f3 - f2.^2).^2 + ( 8*f2.^2.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^3;
Beta23 = (2*f1.*g2)./(f1.*f3 - f2.^2).^2 + (2*f2.*g1)./(f1.*f3 - f2.^2).^2 - ( 4*f1.*f2.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^3;
Beta31 = Beta13;
Beta32 = Beta23;
Beta33 = -(2*f1.*g1)./(f1.*f3 - f2.^2).^2 + ( 2*f1.^2.*(f1.*g3 - 2*f2.*g2 + f3.*g1) )./(f1.*f3 - f2.^2).^3;

Alpha = [Alpha1 Alpha2 Alpha3];
Beta = [Beta11 Beta12 Beta13 Beta21 Beta22 Beta23 Beta31 Beta32 Beta33];

DR = [real(D) -imag(D)];
DI = [imag(D)  real(D)];
DRDI = [DR; DI];

hessian = zeros(4*n);
gradient = zeros(4*n,1);

ss1 = [real(fz)+real(gz);imag(fz)-imag(gz);real(fz)+real(gz);-imag(fz)+imag(gz)].*repmat(Alpha(:,1),4,1);
ss2 = [-imag(gz);-real(gz);-imag(fz);-real(fz)].*repmat(Alpha(:,2),4,1);
ss3 = [real(fz)-real(gz);imag(fz)+imag(gz);-real(fz)+real(gz);imag(fz)+imag(gz)].*repmat(Alpha(:,3),4,1);

gradient(1:2*n)     = 2*DRDI'*(ss1(1:end/2)+ss2(1:end/2)+ss3(1:end/2));
gradient(2*n+1:end) = 2*DRDI'*(ss1(end/2+1:end)+ss2(end/2+1:end)+ss3(end/2+1:end));

%% My method for PSD
HP = Hessian_Beta(Beta(:,[1 2 3 5 6 9]),fz,gz);
HP_A = Hessian_Alpha(Alpha);

HP = HP + 2*HP_A;

D_H = D;
DR_H = [real(D_H) -imag(D_H)];
DI_H = [imag(D_H)  real(D_H)];
DRDI_H = [DR_H; DI_H];

hessian(1:2*n, 1:2*n)             = DRDI_H'*[ DR_H.*HP(1,:)'+DI_H.*HP(5,:)'; DR_H.*HP(2,:)'+DI_H.*HP(6,:)' ];
hessian(end/2+1:end, end/2+1:end) = DRDI_H'*[ DR_H.*HP(11,:)'+DI_H.*HP(15,:)'; DR_H.*HP(12,:)'+DI_H.*HP(16,:)' ];

hessian(1:2*n, end/2+1:end)       = DRDI_H'*[ DR_H.*HP(9,:)'+DI_H.*HP(13,:)'; DR_H.*HP(10,:)'+DI_H.*HP(14,:)' ];
hessian(end/2+1:end, 1:2*n)       = hessian(1:2*n, end/2+1:end)';

RIRI2RRII = [1:n n*2+(1:n) n+(1:n) n*3+(1:n)];
h  = hessian(RIRI2RRII, RIRI2RRII);
g = gradient(RIRI2RRII);
end

function HP = Hessian_Beta(Beta,fz,gz)
    B11 = Beta(:,1);B12 = Beta(:,2);B13 = Beta(:,3);
    B22 = Beta(:,4);B23 = Beta(:,5);B33 = Beta(:,6);
    fzr = real(fz);fzi = imag(fz);
    gzr = real(gz);gzi = imag(gz);
    cprpr = fzr + gzr;
    cprnr = fzr - gzr;
    cpipi = fzi + gzi;
    cpini = fzi - gzi;
    
    HP(:,1) = 4*( B11.*cprpr.^2 + 2*B12.*(-cprpr.*gzi) + 2*B13.*(cprpr.*cprnr) + B22.*gzi.^2 + 2*B23.*(-gzi.*cprnr) + B33.*cprnr.^2 );
    HP(:,2) = 4*( B11.*cprpr.*cpini + B12.*(-cprpr.*gzr - gzi.*cpini) + B13.*(cprpr.*cpipi + cprnr.*cpini) + B22.*gzr.*gzi + B23.*(-gzi.*cpipi - gzr.*cprnr) + B33.*cprnr.*cpipi );
    HP(:,5) = HP(:,2);
    HP(:,6) = 4*( B11.*cpini.^2 + 2*B12.*(-cpini.*gzr) + 2*B13.*(cpini.*cpipi) + B22.*gzr.^2 + 2*B23.*(-gzr.*cpipi) + B33.*cpipi.^2 );
    
    HP(:,11) = 4*( B11.*cprpr.^2 + 2*B12.*(-fzi.*cprpr) + 2*B13.*(-cprnr.*cprpr) + B22.*fzi.^2 + 2*B23.*(fzi.*cprnr) + B33.*cprnr.^2 );
    HP(:,12) = 4*( B11.*(-cpini.*cprnr) + B12.*(-cprpr.*fzr + fzi.*cpini) + B13.*(cprpr.*cpipi + cprnr.*cpini) + B22.*fzr.*fzi + B23.*(-fzi.*cpipi + fzr.*cprnr) + B33.*(-cprnr.*cpipi) );
    HP(:,15) = HP(:,12);
    HP(:,16) = 4*( B11.*cpini.^2 + 2*B12.*(cpini.*fzr) + 2*B13.*(-cpini.*cpipi) + B22.*fzr.^2 + 2*B23.*(-fzr.*cpipi) + B33.*cpipi.^2 );
    
    HP(:,3) = 4*( B11.*cprpr.^2 + B12.*(-cprpr.*fzi - cprpr.*gzi) + B22.*gzi.*fzi + B23.*(gzi.*cprnr - fzi.*cprnr) + B33.* (-cprnr.^2) );
    HP(:,4) = 4*( B11.*cprpr.*(-cpini) + B12.*(-cprpr.*fzr + gzi.*cpini) + B13.*(cprpr.*cpipi - cpini.*cprnr) + B22.*gzi.*fzr + B23.*(-gzi.*cpipi - fzr.*cprnr) + B33.*cprnr.*cpipi );
    HP(:,7) = 4*( B11.*cpini.*cprpr + B12.*(-cpini.*fzi - cprpr.*gzr) + B13.*(-cpini.*cprnr + cprpr.*cpipi) + B22.*gzr.*fzi + B23.*(gzr.*cprnr - fzi.*cpipi) + B33.*(-cprnr).*cpipi );
    HP(:,8) = 4*( B11.*cpini.*(-cpini) + B12.*(-cpini.*fzr + gzr.*cpini) + B22.*fzr.*gzr + B23.*(-gzr.*cpipi - fzr.*cpipi) + B33.*cpipi.^2 );
    
    HP(:,9) = HP(:,3);
    HP(:,10) = HP(:,7);
    HP(:,13) = HP(:,4);
    HP(:,14) = HP(:,8);
    
    HP = HP';
end

function HP_A = Hessian_Alpha(alpha)
    Alpha1 = alpha(:,1);Alpha2 = alpha(:,2);Alpha3 = alpha(:,3);
    
    lambda1 = Alpha1 + Alpha3 + ( (Alpha1 - Alpha3).^2 + Alpha2.^2 ).^(1/2);
    i = find(lambda1<0);
    lambda2 = Alpha1 + Alpha3 - ( (Alpha1 - Alpha3).^2 + Alpha2.^2 ).^(1/2);
    j = find(lambda2<0);
    
    HP_A(1,:) = (Alpha1 + Alpha3)';
    HP_A(3,:) = (Alpha1 - Alpha3)';
    HP_A(4,:) = -Alpha2';
    
    a = (Alpha1 + Alpha3)';b = (Alpha1 - Alpha3)';c = Alpha2';
    HP_A(1,j) = ( a(j) + sqrt(b(j).^2 + c(j).^2) )/2;
    HP_A(3,j) = b(j).*(a(j) + (b(j).^2+c(j).^2).^(1/2))./(b(j).^2+c(j).^2).^(1/2)/2;
    HP_A(4,j) = -c(j).*(a(j) + (b(j).^2+c(j).^2).^(1/2))./(b(j).^2+c(j).^2).^(1/2)/2;
    
    HP_A(1,i) = 0;
    HP_A(3,i) = 0;
    HP_A(4,i) = 0;
    
    HP_A(6,:) = HP_A(1,:);HP_A(11,:) = HP_A(1,:);HP_A(16,:) = HP_A(1,:);
    HP_A(9,:) = HP_A(3,:);HP_A(8,:) = -HP_A(3,:);HP_A(14,:) = HP_A(8,:);
    HP_A(7,:) = HP_A(4,:); HP_A(10,:) = HP_A(4,:);HP_A(13,:) = HP_A(4,:);
end
