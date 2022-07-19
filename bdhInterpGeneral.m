% [x, t, y] = readObj('spiral2.obj');
% [x, t, y] = readObj('data\Bar double twist.obj');
% [x, t, y] = readObj('elephant_skinny_trigulations.obj');
% [y, t, x] = readObj('trol_bdh.obj');
% [x, t, y] = readObj('data\Bar double twist_continusHarmonic.obj');
[x, t, y] = readObj('data\elephant.obj');
% [x, t, y] = readObj('horse.obj');
% [x, t, y] = readObj('raptor.obj');

x = x(:, range(x)>1e-5);

% y = x;

xy0 = {x, y};
t0 = t;

figuredocked; 

% [xy, t] = subdivdision(xy0, t0, nsub);
% [x, y] = deal( xy{:} );

%%
B = findBoundary(x, t);
yb = y(B, :) - repmat(mean(x(B,:)), numel(B), 1);
yb = yb*max(range(x(B,:))./range(yb));
% [x, t] = cdt(x(B, end:-1:1), [], size(x, 1)*2, true); % for double bar twisted (orientation problem)
L = laplacian(x, t, 'cot');
% L = laplacian(x, t, 'random');


%%
nv = size(x, 1);
nf = size(t, 1);
B = findBoundary(x, t);
I = setdiff(1:nv, B);

%%
% yb = y(B, :);
% y = x; y(B, :) = yb;
% y(I, :) = -L(I,I)\L(I,B)*yb;


if ~exist('interpAnchID','var'),    interpAnchID = []; end


% if 1
%     L = laplacian(x, t, 'cot');
%     L = laplacian(x, t, 'cr');
%     L = laplacian(x, t, 'uniform') + laplacian(x, t, 'random')*1e0;
%     y(I, :) = -L(I,I)\L(I,B)*y(B, :);
% end

subplot(221); hm = drawmesh(t, x); set(hm, 'FaceColor', 'none', 'EdgeColor', 'b', 'EdgeAlpha', 0.8);  title('src');
subplot(223); hm = drawmesh(t, y); set(hm, 'FaceColor', 'none', 'EdgeColor', 'b', 'EdgeAlpha', 0.8);  title('dst');
subplot(222); hm = drawmesh(t, x); set(hm, 'FaceColor', 'none', 'EdgeColor', 'b', 'EdgeAlpha', 0.8);  title('SIG13');
subplot(224); hm2 = drawmesh(t, x); set(hm2, 'FaceColor', 'none', 'EdgeColor', 'b', 'EdgeAlpha', 0.8); title('generalized Harmonic Interp with Poisson');
% subplot(235); hm3 = drawmesh(t, x); set(hm3, 'FaceColor', 'none', 'EdgeColor', 'b', 'EdgeAlpha', 0.8); title('generalized Harmonic Interp with bounded distortion');

wts = 0:0.05:1;
% im(:, :, :, numel(wts)) = 0;

[Z, err] = GBDHInterp(x, y, t, wts, struct('boundk', false));
Z2 = metricInterp({x, y}, t, [1-wts; wts]', struct('metric', 'metric tensor', 'hf', -1, 'anchors', interpAnchID));
if ~iscell(Z); Z = {Z}; end
%


for i=1:numel(wts)
    wt = wts(i);
    hm2.Vertices = Z{i};
    hm.Vertices = Z2{i};

    drawnow; pause(.1); % title(sprintf('t=%f', wt));

%     frame = getframe(gcf);
%     if i==1
%         [im, map] = rgb2ind(frame.cdata, 256, 'nodither');
%     else
%         im(:, :, 1, i) = rgb2ind(frame.cdata, map, 'nodither');
%     end
end
% imwrite(im, map, 'elephant_skinny_trigulations_extreme.gif', 'DelayTime', 0.25, 'LoopCount', inf);



% figuredocked; plot(1:size(intErr,1), intErrHarm, ':');
% hold on; plot(1:size(intErr,1), intErr, '-');
% % legend({'harmonic abs err', 'harmonic vec err', 'abs err', 'vec err'});

% e = unique( sort(reshape(t(:, [1 2 2 3 3 1]), [], 2), 2), 'rows' );
% figuredocked; h = plot( [x(e(:,1),1) x(e(:,2),1)]', [x(e(:,1),2) x(e(:,2),2)]' );
% 
% vv2LenErr = abs(abs(vv2e) - abs(vv2e)');
% err = vv2LenErr(sub2ind([nv nv], e(:,1), e(:,2)));


% figuredocked; h = drawmesh(t, y);
% set(h, 'CData', abs(fzbar./fz), 'facecolor', 'flat', 'edgealpha', 0.05);