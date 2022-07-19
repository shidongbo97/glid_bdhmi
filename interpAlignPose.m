function rot_trans = interpAlignPose(anchSrc, anchZ, wt)

if isempty(anchSrc)
    rot_trans = [1; 0];
else
    fNormalize = @(x) x./abs(x);

    rot = 1;
    if size(anchSrc,1)>1 && norm([1 -1]*anchSrc(:,1))>1e-8
        rot = prod( fNormalize([1 -1]*anchSrc).^reshape(wt,1,[]) ) / fNormalize( [1 -1]*anchZ );
    end

%     trans = anchSrc(1,:)*wt - rot*anchZ(1);
    trans = anchSrc(1,1) - rot*anchZ(1);  % fix the first anchor

    rot_trans = [rot; trans];
end
