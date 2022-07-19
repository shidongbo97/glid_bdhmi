total_time = tic;

f_z_prev_dense = DerivativeOfCauchyCoordinatesAtDenseSamples*Phi;
assert(all(f_z_prev_dense ~= 0));
frame_dense = abs(f_z_prev_dense)./f_z_prev_dense;

if(0)
    %active set approach

    abs_f_z_prev_dense = abs(f_z_prev_dense);

    %debug - needs to figure out how to measure the error

    %prev_iso_distortion = max(1./abs_f_z_prev_dense, abs_f_z_prev_dense);

    prev_iso_distortion = max(1./real(f_z_prev_dense.*frame_dense), abs_f_z_prev_dense);

    [peaks, peaks_indices] = findpeaks(prev_iso_distortion);

    %debug - I need to change my code so that the code will identify the first
    %time the algorithm is running. I also need to use a struct with all
    %relevant variables and reset everything that is not relevant - such as tau
    %of other models
    if(~exist('tau'))
        tau = 1;
    end

    upperThreshold = 0.1 + 0.9*max(tau, 1.01); %points with distortion higher than this threshold will be added to the active set (if there is also local max)
    indicesToBeAdded = peaks_indices(find(prev_iso_distortion(peaks_indices) > upperThreshold));
    %lowerThreshold = 0.5 + 0.5*tau; %points with distortion less than this threshold will be removed from the active set
    lowerThreshold = 0.1 + 0.9*tau; %points with distortion less than this threshold will be removed from the active set


    %if(exist('activeSet'))
    if(0)
        activeSet = union(activeSet, indicesToBeAdded);
        indicesToBeRemoved = activeSet(find(prev_iso_distortion(activeSet) < lowerThreshold));
        activeSet = setdiff(activeSet, indicesToBeRemoved);
    else
        activeSet = indicesToBeAdded;
    end

    if(1)
        %figure;
        plot(prev_iso_distortion);
        hold on;
        plot(activeSet, prev_iso_distortion(activeSet), 'o');
        hold off;
    end

    DerivativeOfCauchyCoordinatesAtActiveSamples = [DerivativeOfCauchyCoordinatesAtSamples; DerivativeOfCauchyCoordinatesAtDenseSamples(activeSet, 1:end)];

else %active set is turned off
    DerivativeOfCauchyCoordinatesAtActiveSamples = DerivativeOfCauchyCoordinatesAtSamples;
end

numVirtualVertices = size(CauchyCoordinatesAtP2Phandles, 2);
numSamples = size(DerivativeOfCauchyCoordinatesAtActiveSamples, 1);
numDenseSamples = size(DerivativeOfCauchyCoordinatesAtDenseSamples, 1);




f_z_prev = DerivativeOfCauchyCoordinatesAtActiveSamples*Phi;
assert(all(f_z_prev ~= 0));
frame = abs(f_z_prev)./f_z_prev;



%frame = ones(size(frame)); %reset the frames


%tau = 5;
lambda = 100.0;
%lambda = 0.0;

cvx_begin %quiet
    cvx_solver Mosek

    variable phi(numVirtualVertices, 1) complex
    %variable tau; %real - single global variable
    variable tau(numSamples, 1); %real
    
    %variable s(numSamples, 1); %real - vector
	
    expression fz(numSamples, 1)
    %expression fz_dense(numDenseSamples, 1)

    fz = DerivativeOfCauchyCoordinatesAtActiveSamples*phi;
    %fz_dense = DerivativeOfCauchyCoordinatesAtDenseSamples*phi;

    %minimize tau
    %minimize max(tau)
    %minimize sum(tau)
    %minimize norm(tau, 1)
	minimize norm(tau, 2)
    %minimize max(tau) + lambda*norm(tau, 2)
    %minimize norm(SecondDerivativeOfCauchyCoordinatesAtSmoothnessSamples*phi, 2)
    %minimize lambda*norm(SecondDerivativeOfCauchyCoordinatesAtSmoothnessSamples*phi, 2) / size(SecondDerivativeOfCauchyCoordinatesAtSmoothnessSamples, 1) + norm(tau, 2) / norm(ones(size(tau)), 2)
    
    %ARAP - energy
    %minimize norm(fz_dense.*frame_dense - 1, 2)
    %minimize norm(fz.*frame - 1, 2)
    %minimize norm(fz.*frame - 1, 1)
    
    subject to
        CauchyCoordinatesAtP2Phandles*phi == P2PCurrentPositions;

        standard_cones = 0;
        linear_constraints_only = 0;
        arap = 0;
        
        if(standard_cones)
            if(linear_constraints_only)
                rho = sqrt(2)/2;

                tau - s <= rho*(s + tau);
                tau - s >= -rho*(s + tau);
                2 <= rho*(s + tau);
                2 >= -rho*(s + tau);
                
                s <= real(fz.*frame);

                real(fz) <= rho*tau;
                real(fz) >= -rho*tau;
                imag(fz) <= rho*tau;
                imag(fz) >= -rho*tau;

                
            else
                abs(tau - s + 2i) <= s + tau;
                s <= real(fz.*frame);
                abs(fz) <= tau;
            end
        else
            if(arap)
                1-tau <= real(fz.*frame);
                abs(fz) <= 1+tau;
                %0.05 <= real(fz.*frame);
            else
                inv_pos(tau) <= real(fz.*frame);
                abs(fz) <= tau;

                %0.1 <= real(fz.*frame);
            end
            
        end
    
%         rho = sqrt(2)/2;
%         real(fz) <= rho*tau;
%         real(fz) >= -rho*tau;
%         imag(fz) <= rho*tau;
%         imag(fz) >= -rho*tau;
        

%         abs(real(fz)) <= sqrt(2)/2*tau;
%         abs(imag(fz)) <= sqrt(2)/2*tau;

cvx_end

evalutation_time = tic;

fz_dense = DerivativeOfCauchyCoordinatesAtDenseSamples*phi;
abs_fz_dense = abs(fz_dense);
abs_fz = abs(fz);
min_real_dense = min(real(fz_dense.*frame_dense));

%     fprintf('Bounds sparse: 1/tau: %.3f, min_real: %.3f, min|fz|: %.3f, max|fz|: %.3f, tau: %.3f.\n', 1/tau, min(real(fz.*frame)), min(abs_fz), max(abs_fz), tau);
%     fprintf('Bounds  dense: 1/tau: %.3f, min_real: %.3f, min|fz|: %.3f, max|fz|: %.3f, tau: %.3f.\n', 1/tau, min_real_dense, min(abs_fz_dense), max(abs_fz_dense), tau);

toc(evalutation_time);

if(min_real_dense > 1e-5) %the map is probably injective (we use very dense sampling but in order to be 100% sure we still need infinite number of samples)
    injectivity_status = 'Injective';
else
    injectivity_status = 'Non-injective';
end

if(strcmp(injectivity_status, 'Injective') && (strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')))
    Phi = phi;
end

fprintf('%s map, CVX status: %s, Samples: %d, Vars: %d, Time: %.3f sec, Energy: %.3e \n', injectivity_status, cvx_status, numSamples, numVirtualVertices, toc(total_time), cvx_optval);

% norm(SecondDerivativeOfCauchyCoordinatesAtSmoothnessSamples*phi, 2) / size(SecondDerivativeOfCauchyCoordinatesAtSmoothnessSamples, 1)
% norm(tau, 2) / norm(ones(size(tau)), 2)
