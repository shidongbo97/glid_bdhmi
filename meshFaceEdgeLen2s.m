function l = meshFaceEdgeLen2s(x, t)

if ~isreal(x),    x = [real(x) imag(x)]; end

frownorm2 = @(M) sum(M.^2, 2);

l = frownorm2( x(t(:, [2 3 1]), :) - x(t(:, [3 1 2]), :) );
l = reshape( l, [], 3 );

% l = reshape( sum((x(t(:, [2 3 1]), :) - x(t(:, [3 1 2]), :)).^2, 2), [], 3 );