function [fdif, fzdif, fzbardif, f, fz, fzbar] = fDifOnMeshEdges(cage, x, e, wt, phi, psy, g, alpha)

if hasGPUComputing
    if ~isa(gpuArray(cage), 'gpuArray'), cage = gpuArray(cage); end
    if ~isa(gpuArray(x), 'gpuArray'),    x = gpuArray(x); end
    if ~isa(gpuArray(wt), 'gpuArray'),   wt = gpuArray(wt); end
    if ~isa(gpuArray(phi), 'gpuArray'),  phi = gpuArray(phi); end
    if ~isa(gpuArray(psy), 'gpuArray'),  psy = gpuArray(psy); end
    if ~isa(gpuArray(g), 'gpuArray'),    g = gpuArray(g); end
end
    


ne = size(e, 1);
nx = numel(x);
MV2Evec = sparse( repmat(1:ne, 2, 1)', e, repmat( [-1 1], ne, 1), ne, nx );
% fz = derivativesOfCauchyCoord(cage, x)*phi;
% fzr = fz(e(:,2))./fz(e(:,1));
% thetaDiffs = angle( fzr );
% gdif = complex(log(abs(fzr)), thetaDiffs);
% g2 = [MV2Evec; sparse(1, 1, 1, 1, nx)]\[gdif; log(derivativesOfCauchyCoord(cage,x(1))*phi)];
% norm(g2-g)
% g2 = logFzWithRef(derivativesOfCauchyCoord(cage, x)*phi, g);

fprintf('Computing map with numerical integral ... ');
tic;

% fzdif = zeros(ne, 1);
% fzbardif = zeros(ne, 1);
% for i=1:ne
%     fzdif(i) = integral(@(z) fzt(cage, z, phi, wt, g(e(i,1))), x(e(i,1)), x(e(i,2)));
%     fzbardif(i) = integral(@(z) fzbart(cage, z, phi, psy, wt, g(e(i,1))), x(e(i,1)), x(e(i,2)));
%     
% %     fzbardif(i) = integral(@(z) conj(fzbart(cage, z, phi, psy, wt, g(e(i,1)))), x(e(i,1)), x(e(i,2)));
% %     fdif(i) = fzdif(i) + conj(fzbardif(i));
% end

[fzdif, fzdif2] = gkq(@(z) fzt(cage, z, phi, wt, g(e(:,1))), x(e(:,1)), x(e(:,2)));
[fzbardif, fzbardif2] = gkq(@(z) fzbart(cage, z, phi, psy, wt, g(e(:,1)), alpha), x(e(:,1)), x(e(:,2)));
fdif = fzdif + conj(fzbardif);

% [max( fzbardif2' ) max( fzdif2' )]
fprintf('time: %.5f\n', toc);

f1 = MV2Evec\gather(fzdif);
f2 = MV2Evec\gather(fzbardif);
fprintf('dif = %f, %f\n', norm(MV2Evec*f1-fzdif), norm(MV2Evec*f2-fzbardif));
f = f1 + conj(f2);
% f = MV2Evec\gather(fdif);

fz = exp(wt*g);
D = derivativesOfCauchyCoord(cage, x);
fzbar = wt^alpha*D*psy.*( (D*phi)./exp(wt*g) );

function r = fzt(cage, z, phi, wt, g0)

g = logFzWithRef(derivativesOfCauchyCoord(cage, z)*phi, g0);
r = reshape(exp(wt*g), size(z));

function r = fzbart(cage, z, phi, psy, wt, g0, alpha)

D = derivativesOfCauchyCoord(cage, z);
g = logFzWithRef(D*phi, g0);
% r = reshape(wt*D*psy.*conj( exp(wt*g)./(D*phi) ), size(z)); % conj make integral problematic
% r = reshape(wt*D*psy.*( exp(wt*g)./(D*phi) ), size(z)); % conj make integral problematic
r = reshape(wt^alpha*D*psy.*( (D*phi)./exp(wt*g) ), size(z)); % conj make integral problematic
% r = conj(fzbart) = conj(fzbar)*fz/fzt;


function g = logFzWithRef(fz, g0)

fzr = fz./exp(g0);
g = g0 + complex(log(abs(fzr)), angle(fzr));
