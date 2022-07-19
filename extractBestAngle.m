function [theta, thetaDiffs] = extractBestAngle(fz)
%given fz on the boundary of a polygon (at a set of samples) we compute the angle theta at each sample such that exp(1i*theta)=exp(1i*angle(fz)).
%note that adding 2*pi*k to each sample (where k is an integer) doesn't affect the correctness of the above equation.
%however, we would like to set k at each sample automatically such that the difference between each two consecutive angles is minimized.

    thetaDiffs = angle(fz./fz([end 1:end-1]));
    theta = angle(fz(end)) + cumsum(thetaDiffs);
    thetaDiffs(1) = theta(1) - theta(end);
%     %sanity check
%     err = norm(exp(1i*theta)-exp(1i*angle(fz)), Inf);
%     assert(err < 1e-10);
    
end


% 
% function [theta, maxAbsAngleDifference] = renjie_extractBestAngle(fz)
% %given fz on the boundary of a polygon (at a set of samples) we compute the angle theta at each sample such that exp(1i*theta)=exp(1i*angle(fz)).
% %note that adding 2*pi*k to each sample (where k is an integer) doesn't affect the correctness of the above equation.
% %however, we would like to set k at each sample automatically such that the difference between each two consecutive angles is minimized.
% 
%     assert(all(fz ~= 0));
% 
%     cyclicDiff = @(x) x - circshift(x, 1);
%     mylog = @(x, r) log( x*exp(1i*r) ) - 1i*r;
% 
%     %thetaDiffs = cyclicDiff( [log(fz) mylog(fz, pi/2) mylog(fz, -pi/2) mylog(fz, pi) mylog(fz, -pi)] );
%     thetaDiffs = cyclicDiff( [log(fz) mylog(fz, pi)] );
%     
%     [~, index] = min(abs( thetaDiffs ), [], 2);
%     theta = imag(log(fz(end)) + cumsum( thetaDiffs( sub2ind(size(thetaDiffs), (1:size(thetaDiffs,1))', index) ) ));
% 
%     maxAbsAngleDifference = max(abs(cyclicDiff(theta)));
%     
%     
%     %change this function to use angle rather than complex log. more efficient?
%     %change mylog to use cleaer notations. r is an angle? seems like modulus
%     
%     %sanity check
%     err = norm(exp(1i*theta)-exp(1i*angle(fz)), Inf);
%     assert(err < 1e-9);
%     
% end


% function [theta, maxAbsAngleDifference] = extractBestAngle(fz)
% %given fz on the boundary of a polygon (at a set of samples) we compute the angle theta at each sample such that exp(1i*theta)=exp(1i*angle(fz)).
% %note that adding 2*pi*k to each sample (where k is an integer) doesn't affect the correctness of the above equation.
% %however, we would like to set k at each sample automatically such that the difference between each two consecutive angles is minimized.
% 
%     assert(all(fz ~= 0));
% 
%     complexDot = @(z1, z2) real(z1.*conj(z2)); %dot product
%     complexCross = @(z1, z2) imag(conj(z1).*z2); %cross product
%     cyclicDiff = @(x) x - circshift(x, 1);
%  
%     a = fz; %current
%     b = circshift(fz, 1); %prev
%     
%     cos = complexDot(a, b)./abs(a.*b);
%     cos(cos > 1) = 1;
%     thetaDiffs = acos(min(cos, ones));
%     reversedOrder = find(complexCross(b, a) < 0);
%     thetaDiffs(reversedOrder) = -thetaDiffs(reversedOrder);
%     
%     theta = angle(fz(end)) + cumsum(thetaDiffs);
%     
%     principalAngles = angle(fz);
%     branchIndex = round((theta - principalAngles)/(2*pi));
%     theta = principalAngles + 2*pi*branchIndex;
%     
%     maxAbsAngleDifference = max(abs(cyclicDiff(theta)));
%     
%     %sanity check
%     err = norm(exp(1i*theta)-exp(1i*angle(fz)), Inf);
%     assert(err < 1e-10);
%     
%     
% %     [renjie_theta, renjie_maxAbsAngleDifference] = renjie_extractBestAngle(fz);
% %     assert(norm(renjie_theta-theta, Inf) < 1e-5);
% end
% 

