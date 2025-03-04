function [E, R, t] = estimate_E_R_with_t_constraint(points1, points2, intrinsics)
    % ESTIMATE_R_WITH_T_CONSTRAINT Estimates the rotation matrix (R) and translation vector (t) 
    % from matched stereo image points under a planar motion constraint.
    %
    % Description:
    %   This function computes the essential matrix (E) from stereo image calibration points 
    %   and decomposes it into possible rotation matrices (R) and a constrained translation 
    %   vector (t), assuming that cameras are placed with no vertical displacement (tz = 0).
    %   The function selects the best R based on triangulated 3D point errors and refines R 
    %   using nonlinear optimization (Levenberg-Marquardt method).
    %
    % Inputs:
    %   points1:      (Nx2) matrix of feature points from camera 1 (x, y).
    %   points2:      (Nx2) matrix of corresponding feature points from camera 2 (x, y).
    %   intrinsics:   Camera intrinsics object containing the intrinsic matrix K:
    %                 K = [fx,  0, cx;
    %                      0,  fy, cy;
    %                      0,   0,  1];
    %                 where fx, fy are focal lengths (in pixels), and cx, cy are principal 
    %                 points (in pixels). cx = imageWidth(pixels) / 2, cy = imageHeight(pixels) / 2
    %
    % Outputs:
    %   R:            (3x3) Rotation matrix aligning the two camera views.
    %   t:            (3x1) Norm translation vector with enforced planar constraint (tz = 0).
    %
    % Example Usage:
    %   [R, t] = estimate_R_with_t_constraint(matched_points1, matched_points2, cameraIntrinsics);
    %
    % Notes:
    %   - The function assumes cameras are at approximately the same height, enforcing tz = 0.
    %   - The best rotation matrix R is selected based on the number of triangulated points with 
    %     positive depth. Similar R1 and R2 matrices could cause the
    %     algorithm to select the wrong one.
    %   - R is further refined using Levenberg-Marquardt nonlinear optimization to minimize 
    %     reprojection error.
    %
    % Nolan R Bonnie, 01/2025
    % nolan.bonnie@colorado.edu


    % Estimate the rotation matrix R and translation matrix t given real
    % world calibration points, intrinsic camera values (K = [fx,  0, cx; 0, fy, cy; 0,  0,  1];)
    % Where fx, fy are the focal lengths in pixels, and cx, cy are the
    % principal points in pixels (cx = imageWidth(pixels) / 2, cy = imageHeight(pixels) / 2)
    % K should be calibrated for every camera body + lens focal length pair
    % manually with calibration boards in a controlled setting. Use only
    % prime lenses. 

    
    % Step 1: Estimate Essential Matrix from calibration points and
    % intrinsic camera values
    E = estimateEssentialMatrix(points1, points2, intrinsics);

    % Step 2: Extract Rotation and Translation Matrices
    % Decompose E into R and t (two possible solutions for R)
    [R1, R2, t1, t2] = decomposeEssentialMatrix(E); % use decomposeEssentialMatrix(E, true) for more verbose information

    % Step 3: Apply our constraint that  t = [tx, ty, 0]  (assuming tz ~ 0)
    % There's an assumption here that cameras are placed with approximately the same height off the ground
    t1(3) = 0; % Set tz = 0  
    t1 = t1 / norm(t1); % Re-normalize t (we want norm t = 1, (aka 1 meter), so we can scale by real world units easily later, but this step is arbitrary)

    t2(3) = 0;
    t2 = t2 / norm(t2);

    % Step 4: triangulate the calibration points to pick the better R estimate
    % P1 & P2 are the projection matrixes of each camera
    K = intrinsics.K;
    P1 = K * [eye(3), zeros(3,1)];
    P2_1 = K * [R1, t1];
    P2_2 = K * [R2, t2];

    points3D_1 = triangulate(points1, points2, P1, P2_1);
    points3D_2 = triangulate(points1, points2, P1, P2_2);

    % Compute bounding box for both reconstructions
    min1 = min(points3D_1, [], 1);
    max1 = max(points3D_1, [], 1);
    min2 = min(points3D_2, [], 1);
    max2 = max(points3D_2, [], 1);
    
    % Calculate bounding box volumes
    volume1 = prod(max1 - min1);
    volume2 = prod(max2 - min2);

    % Choose R based on the smaller bounding box volume
    % Logic here is that bad R calculations create situations where points
    % explode on the axes, and exist far outside the real volume
    if volume1 <= volume2
        R = R1;
        t = t1;
    else
        R = R2;
        t = t2;
    end
    
    % Adding in a manual user override that can swap the logic to grab the
    % opposite R matrix. This should never be required. If so- revisit the
    % R selection criteria
    override = false;
    if override == true
        if volume1 > volume2
            R = R1;
            t = t1;
        else
            R = R2;
            t = t2;
        end
    end


    % Step 5: Refine R using nonlinear optimization
    % This could theoretically refine the accuracy of R if not already at
    % the optimal calculation. Bad calibration points might cause errors. 

    R_vec = rotationMatrixToVector(R); % Convert to Rodrigues vector

    % Optimize using the Levenberg-Marquardt / Gauss-Newton least squares non-linear method
    % Tune optimization parameters: https://www.mathworks.com/help/releases/R2024b/optim/ug/lsqnonlin.html#lsqnonlin_opts
    options = optimoptions(@lsqnonlin, ...      % Function to apply these options to
        'MaxFunctionEvaluations', 1000, ...     % Maximum number of function evaluations allowed
        'MaxIterations', 500, ...               % Maximum number of iterations allowed
        'TolFun', 1e-6, ...                     % Set function tolerance
        'TolX', 1e-6, ...                       % Set variable tolerance
        'InitDamping', 1, ...                   % Set an initial damping factor
        'ScaleProblem', 'none', ...             % 'jacobian' can sometimes improve the convergence of a poorly scaled problem
        'UseParallel', false, ...               % Could parallel process- takes ~10s to connect to parallel resources (not worth it right now)
        'Display', 'none', ...                  % iteration details set to 'iter-detailed' to debug
        'Algorithm', 'levenberg-marquardt');    % Specify the algorithm used 'levenberg-marquardt'    

    R_opt = lsqnonlin(@(r) reprojectionError(r, t, points1, points2, K), R_vec, [], [], options);
    R = rotationVectorToMatrix(R_opt); % Convert back to rotation matrix

end


function [R1, R2, t1, t2] = decomposeEssentialMatrix(E, varargin)
    % Step 1: Compute SVD of Essential Matrix
    [U, ~, V] = svd(E);
    
    % Ensure determinant is positive to maintain a valid rotation matrix
    if det(U) < 0, U = -U; end
    if det(V) < 0, V = -V; end
    
    % Step 2: Define the special W matrix
    W = [0 -1 0; 1 0 0; 0 0 1];

    % Step 3: Compute Two Possible Rotations
    R1 = U * W * V';
    R2 = U * W' * V';

    % Ensure R1 and R2 are valid rotation matrices (det(R) should be +1)
    if det(R1) < 0, R1 = -R1; end
    if det(R2) < 0, R2 = -R2; end

    % Step 4: Extract Two Possible Translations
    t1 = U(:,3);
    t2 = -U(:,3);  % Translation is defined up to scale
    
    if isscalar(varargin)
            % Display results for debugging
            disp('Possible Rotation Matrices:');
            disp('R1 = '); disp(R1);
            disp('R2 = '); disp(R2);
            disp('Possible Translation Vectors:');
            disp('t1 = '); disp(t1);
            disp('t2 = '); disp(t2);
    end
end


function err = reprojectionError(r, t, points1, points2, K)
    R = rotationVectorToMatrix(r);
    P1 = K * [eye(3), zeros(3,1)];
    P2 = K * [R, t];

    % Triangulate 3D points
    points3D = triangulate(points1, points2, P1, P2);

    % Project back to image planes
    proj1 = projectPoints(points3D, P1);
    proj2 = projectPoints(points3D, P2);

    % Compute error as reprojection difference
    err = [points1(:) - proj1(:); points2(:) - proj2(:)];
end


function proj = projectPoints(points3D, P)
    % Projects 3D points into the 2D image plane
    points3D_h = [points3D, ones(size(points3D, 1), 1)]'; % Homogeneous coords
    proj_h = P * points3D_h; % Projected points
    proj = (proj_h(1:2,:) ./ proj_h(3,:))'; % Convert to 2D
end



% OLD METHOD: Essential Matrix calculation unreliable due to the
% complexity of optimizing F such that points1 * F * points2' = 0 for
% all points. 
%
%
% % Step 1: Compute Essential Matrix
% E = K' * F * K;
% 
% % Step 2: Enforce Rank-2 Constraint on E, since E = [t]xR is rank 2, S
% % should be rank 2 as well. May already be satisfied.
% [U, S, V] = svd(E);
% S(3,3) = 0;
% E = U * S * V';