function x = alignPoseByAnchors(x, anchorID, xAnchor, wt)

r_t = interpAlignPose(xAnchor, x(anchorID), wt);

x = x*r_t(1) + r_t(2);