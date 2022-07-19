function k = kFromFaceEdgeLen2s(xlen2s, ylen2s)

r = sqrt( (sum(xlen2s, 2).^2 - 2*sum(xlen2s.^2,2)) .* (sum(ylen2s, 2).^2 - 2*sum(ylen2s.^2,2)) ) ./ ( sum(xlen2s, 2).*sum(ylen2s, 2) - 2*dot(xlen2s, ylen2s, 2) );
k = sqrt( max(1-r, 0) ./ (1+r) );