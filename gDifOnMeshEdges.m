function [gdif, e] = gDifOnMeshEdges(cage, x, t, phi)
%% compute differences of g on all the edges of the triangulation (x, t)

e = [t(:) reshape(t(:, [2 3 1]), [], 1)];
e = unique( sort(e, 2), 'rows' );

fprintf('Computing integral logfz''=g'' on all edges ... ');

fz = derivativesOfCauchyCoord(cage, x)*phi;

%% g = log(fz)
% gdif2 = log( fz(e(:,2)) ) -  log( fz(e(:,1)) );

%% g = smarter version of log(fz)
fzr = fz(e(:,2))./fz(e(:,1));
thetaDiffs = angle( fzr );
gdif3 = complex(log(abs(fzr)), thetaDiffs);

%% use numerical integral to compute g
tic;
% gdif = zeros(size(e,1), 1);
% % parfor
% for i=1:size(e,1)
%     gdif(i) = integral(@(z) derivativeOfLog_fz(cage, z, phi), x(e(i,1)), x(e(i,2)));
% end

gdif = gdif3;
fprintf('time: %.5f, dif = %f\n', toc, norm(gdif3-gdif));

function r = derivativeOfLog_fz(cage, z, phi)
% z can be a scalar or vector of complex points

[D, E] = derivativesOfCauchyCoord(cage, z);
r = (E*phi)./(D*phi);
% r = gather(r.');
r = r.';
