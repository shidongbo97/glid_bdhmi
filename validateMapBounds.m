function [phi_valid, psy_valid, t] = validateMapBounds(L, indexOfMaxL, v, fillDistanceSegments, phi_prev, phi_next, psy_prev, psy_next, DerivativeOfCauchyCoordinates, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, maxBisectionIterations, unrefinedInwardCage, use_Cauchy_argument_principle)
%given a previous solution (phi_prev, psy_prev) and a new solution (phi_next, psy_next), this function checks whether the new solution
%is valid (satisfy the requiered bounds). The function assumes that the previous solution is valid.
%if the new solution is not valid a line search is being performed until a valid solution is reached.
%   phi_valid, psy_valid - the valid solutions
%   t - a parameter between 0 and 1. t=0 means the prev solution is being returned.
%       t=1 means the next solution is valid and is being returned.
%       any value smaller than 1 indicates that the obtained valid solution was found on the boundary of the feasible domain.
%
%   inwardOffsetCage - can be the inward offset cage without any virtual vertices (for better speed)

%debug - right now I don't search the upper part of the space for a solution. this actually means that the obtained solution will not be on the boundary of the feasible domain.
%I think that this way there is less chance to get stuck


if iscell(v) && numel(v)==1, v = v{1}; end

assert(~iscell(v), 'multi-connected domain is not supported yet');


%debug - maybe expose this to the user?
sigma2_min_allowed = 0.7*sigma2_lower_bound;
sigma1_max_allowed = 1.3*sigma1_upper_bound;
k_max_allowed = min(1, 1.2*k_upper_bound);

t = 1;

for i=1:maxBisectionIterations
    
    phi_valid = (1-t)*phi_prev + t*phi_next;
    psy_valid = (1-t)*psy_prev + t*psy_next;
    
    %return; %for debugging - this disables the validation
    
    if hasGPUComputing
        [fz_does_not_vanish, lowerBoundGlobal_sigma2, upperBoundGlobal_sigma1, upperBoundGlobal_k, minOnSamples_sigma2, maxOnSamples_sigma1, maxOnSamples_k] = computeBoundsOnAllTypeOfDistortion(L, indexOfMaxL, v, fillDistanceSegments, DerivativeOfCauchyCoordinates, gpuArray(phi_valid), gpuArray(psy_valid), unrefinedInwardCage, use_Cauchy_argument_principle);
    else
        [fz_does_not_vanish, lowerBoundGlobal_sigma2, upperBoundGlobal_sigma1, upperBoundGlobal_k, minOnSamples_sigma2, maxOnSamples_sigma1, maxOnSamples_k] = computeBoundsOnAllTypeOfDistortion(L, indexOfMaxL, v, fillDistanceSegments, DerivativeOfCauchyCoordinates, phi_valid, psy_valid, unrefinedInwardCage, use_Cauchy_argument_principle);
    end
    
    fprintf('t: %.5f, sigma2: (%.3f, %.3f, %.3f), sigma1: (%.3f, %.3f, %.3f), k: (%.3f, %.3f, %.3f)\n', ...
        t, ...
        sigma2_lower_bound, minOnSamples_sigma2, lowerBoundGlobal_sigma2, ...
        sigma1_upper_bound, maxOnSamples_sigma1, upperBoundGlobal_sigma1, ...
        k_upper_bound, maxOnSamples_k, upperBoundGlobal_k);
    
    if (fz_does_not_vanish && ...
            lowerBoundGlobal_sigma2 >= sigma2_min_allowed && ...
            upperBoundGlobal_sigma1 <= sigma1_max_allowed && ...
            upperBoundGlobal_k <= k_max_allowed)
        
        return; %everything is ok. map is locally injective with all bounds satisfied
    else
        
        fprintf('Map corresponding to t: %.5f is invalid. Trying to reduce t...\n', t);
        if(i == maxBisectionIterations-1) %last iteration
            fprintf('Line search failed to progress. Reverting to previous map (t=0)\n');
            t = 0;
        else
            t = t/2;
        end
    end
end
%
%
%     %if we reached here it means that we didn't manage to find a valid solution after maxBisectionIterations iterations
%     t = 0;
%     phi_valid = phi_prev;
%     psy_valid = psy_prev;

end



function [fz_does_not_vanish, lowerBoundGlobal_sigma2, upperBoundGlobal_sigma1, upperBoundGlobal_k, ...
    minOnSamples_sigma2,     maxOnSamples_sigma1,     maxOnSamples_k] = ...
    computeBoundsOnAllTypeOfDistortion(L, indexOfMaxL, v, fillDistanceSegments, DerivativeOfCauchyCoordinates, phi, psy, unrefinedInwardCage, use_Cauchy_argument_principle)

tic;

fz = DerivativeOfCauchyCoordinates*phi;

[~, thetaDiffs] = extractBestAngle(fz);

abs_fz = abs(fz);

[L_fz, L_fzbar] = computeLipschitzConstantOf_fz_and_fzbar(L, indexOfMaxL, v, phi, psy);
%[L_fz, L_fzbar] = computeLipschitzConstantOf_fz_and_fzbar_simple(L, indexOfMaxL, v, phi, psy);

fprintf('L_fz: %.2f, L_fzbar: %.2f\n', max(L_fz), max(L_fzbar));

deltaTheta = abs(thetaDiffs);

%this is the sufficient condition to assure that fz does not vanish inside the domain
if(all(1e-5 + (2 + deltaTheta).*L_fz.*fillDistanceSegments < (2 - deltaTheta).*(abs_fz + abs_fz([2:end 1]))))
    
    fz_does_not_vanish = true;
else %sufficient conditions failed but there is still a chance that fz does not vanish since this is not a necessary condition

    if(use_Cauchy_argument_principle)
        %we use the Cauchy argument principle to get a sharp answer to whether fz vanishes or not

        fprintf('\n');
        numZeros = computeNumOfZerosOf_fz_inside(v, phi, unrefinedInwardCage);
        fprintf('numZeros of fz inside the domain: %d.\n', numZeros);
        if(numZeros == 0)
            fz_does_not_vanish = true;
        else
            fz_does_not_vanish = false;
        end
        fprintf('\n');
    else
        fz_does_not_vanish = false;
    end
end

if(fz_does_not_vanish)
    %now that we know that fz does not vanish in the domain we can bound sigma2, sigma1, and k.
    upperBoundOnEachSegment_abs_fz = computeUpperBounds(L_fz, fillDistanceSegments, abs_fz);

    abs_fzbar = abs(DerivativeOfCauchyCoordinates*psy);
    
    upperBoundOnEachSegment_abs_fzbar = computeUpperBounds(L_fzbar, fillDistanceSegments, abs_fzbar);
    lowerBoundOnEachSegment_abs_fz = computeLowerBounds(L_fz, fillDistanceSegments, abs_fz);
    
    lowerBoundGlobal_sigma2 = min(lowerBoundOnEachSegment_abs_fz - upperBoundOnEachSegment_abs_fzbar);
    upperBoundGlobal_sigma1 = max(upperBoundOnEachSegment_abs_fz + upperBoundOnEachSegment_abs_fzbar);
    upperBoundGlobal_k = max(upperBoundOnEachSegment_abs_fzbar ./ lowerBoundOnEachSegment_abs_fz);
    
    minOnSamples_sigma2 = min(abs_fz - abs_fzbar);
    maxOnSamples_sigma1 = max(abs_fz + abs_fzbar);
    maxOnSamples_k = max(abs_fzbar ./ abs_fz);
    
    % 	lowerBoundGlobal_sigma2_notTight = lowerBoundGlobal_abs_fz - upperBoundGlobal_abs_fzbar;
    %     upperBoundGlobal_sigma1_notTight = upperBoundGlobal_abs_fz + upperBoundGlobal_abs_fzbar;
    %     upperBoundGlobal_k_notTight = upperBoundGlobal_abs_fzbar / lowerBoundGlobal_abs_fz;
    
else %fz vanishes

    fprintf('fz vanishes inside the domain!\n');
    fz_does_not_vanish = false;
    lowerBoundGlobal_sigma2 = -Inf;
    minOnSamples_sigma2 = -Inf;
    upperBoundGlobal_sigma1 = Inf;
    maxOnSamples_sigma1 = Inf;
    upperBoundGlobal_k = Inf;
    maxOnSamples_k = Inf;
end

fprintf('computeBoundsOnAllTypeOfDistortion time: %.5f\n', toc);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   the following two functions compute a lower or upper bounds for a real function f on a polygon.
%
%   L_segments - the Lipschitz constant of the function f on each segment.
%   fillDistanceSegments - length of segments divided by 2.
%   f - the known values of the function (must be real) at the samples (vertices of the polygon).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lowerBoundOnEachSegment, lowerBoundGlobal] = computeLowerBounds(L_segments, fillDistanceSegments, f)

avg_val = 1/2*(f + f([2:end 1]));

lowerBoundOnEachSegment = avg_val - L_segments.*fillDistanceSegments;

if(nargout > 1)
    lowerBoundGlobal = min(lowerBoundOnEachSegment);
end
end


function [upperBoundOnEachSegment, upperBoundGlobal] = computeUpperBounds(L_segments, fillDistanceSegments, f)

avg_val = 1/2*(f + f([2:end 1]));

upperBoundOnEachSegment = avg_val + L_segments.*fillDistanceSegments;

if(nargout > 1)
    upperBoundGlobal = max(upperBoundOnEachSegment);
end
end



function [L_Re_fz_exp_i_theta] = computeLipschitzConstantOf_Re_fz_exp_i_theta(L_fz, theta, upperBoundOnEachSegment_abs_fz, fillDistanceSegments)

%|theta2-theta1| / |v2-v1|
L_exp_i_theta = 2*abs((theta([2:end 1]) - theta)) ./ fillDistanceSegments;

L_Re_fz_exp_i_theta = L_fz + L_exp_i_theta.*upperBoundOnEachSegment_abs_fz;

end
