function result = stereoReconstruct3D(input_path, distance_between_camera_meters)
% stereoReconstruct3D Production-ready 3D reconstruction from stereo data.
%
%   result = stereoReconstruct3D(input_path, distance_between_camera_meters)
%
%   This function takes the path to a folder that contains two subfolders,
%   'cam1' and 'cam2'. Each subfolder should contain one xyt file that can be 
%   a .mat or .csv file with an xyt array (and other required fields).
%   The second input is the distance between cameras in meters. It processes 
%   the data, performs calibration, matching, triangulation, and trajectorizing,
%   and returns a struct with the results.
%
%   The output struct includes:
%       - xyzt       : Triangulated 3D points with timestamp.
%       - xyztkj     : Processed trajectories.
%       - input_path : The input folder path.
%       - intrinsics : Camera intrinsic parameters.
%       - K          : Camera intrinsic matrix K.
%       - E          : Essential matrix (if applicable).
%       - R          : Rotation matrix.
%       - t          : Translation vector.
%       - F          : Fundamental matrix (if applicable).
%       - calTraj    : Calibration trajectories.
%       - dk         : Frame delta.
%       - frame_width: Frame width.
%       - frame_height: Frame height.
%       - calculation_method: "E" (Essential) or "F" (Fundamental)
%
%   Example:
%       result = stereoReconstruct3D('/Users/home/.../<YEAR>/<LOCATION>/<DATE>/Sony_A7_xyt/', 0.9906);

    %% Initialization and Input Validation
    logMessage('Starting stereoReconstruct3D...');
    
    if ~isfolder(input_path)
        error('Input path does not exist: %s', input_path);
    end
    if ~isscalar(distance_between_camera_meters) || distance_between_camera_meters <= 0
        error('Distance between cameras must be a positive scalar.');
    end
    logMessage(sprintf('Input path: %s', input_path));
    %logMessage(sprintf('Distance between cameras: %f', distance_between_camera_meters));
    
    %% Define subfolder paths for cam1 and cam2
    cam1_folder = fullfile(input_path, 'cam1');
    cam2_folder = fullfile(input_path, 'cam2');
    
    if ~isfolder(cam1_folder) || ~isfolder(cam2_folder)
        error('Both cam1 and cam2 subfolders must exist in the input path.');
    end
    %logMessage(sprintf('Found cam1 folder: %s', cam1_folder));
    %logMessage(sprintf('Found cam2 folder: %s', cam2_folder));
    
    %% Read and Load Files from Subfolders
    % Gather files with .mat and .csv extensions
    files_cam1 = [dir(fullfile(cam1_folder, '*.mat')); ...
                  dir(fullfile(cam1_folder, '*.csv'))];
    files_cam2 = [dir(fullfile(cam2_folder, '*.mat')); ...
                  dir(fullfile(cam2_folder, '*.csv'))];
              
    if isempty(files_cam1)
        error('No .mat or .csv files found in cam1 folder: %s', cam1_folder);
    end
    if isempty(files_cam2)
        error('No .mat or .csv files found in cam2 folder: %s', cam2_folder);
    end
    
    %logMessage(sprintf('Found %d file(s) in cam1 folder.', numel(files_cam1)));
    %logMessage(sprintf('Found %d file(s) in cam2 folder.', numel(files_cam2)));
    
    % Load the first file from each subfolder using the helper function.
    [df1, c1n, ff1] = loadXytFromFile(fullfile(cam1_folder, files_cam1(1).name));
    [df2, c2n, ~] = loadXytFromFile(fullfile(cam2_folder, files_cam2(1).name));
    logMessage(sprintf('Loaded file from cam1: %s', files_cam1(1).name));
    logMessage(sprintf('Loaded file from cam2: %s', files_cam2(1).name));
    
    %% Determine Frame Dimensions
    % Try to read frame dimensions from the ff field (if available)
    try
        if ~isempty(ff1) && isfield(ff1, 'bkgrStack')
            frame_width = size(ff1.bkgrStack, 2);
            frame_height = size(ff1.bkgrStack, 1);
            %logMessage('Frame dimensions obtained from file.');
        else
            error('ff field not available');
        end
    catch
        frame_width = 1920;
        frame_height = 1080;
        logMessage('Warning: Movie information not read in from files. Using default frame dimensions (1920x1080).');
    end
    
    %% Adjust Y-Coordinates
    max_y = max([df1(:,2); df2(:,2)]);
    df1(:,2) = max_y - df1(:,2);
    df2(:,2) = max_y - df2(:,2);
    %logMessage('Adjusted y-values by flipping based on maximum y-value.');
    
    %% Add Additional Functions Path
    % Adjust this path as needed for your production environment.
    functions_path = fullfile(fileparts(mfilename('fullpath')), 'main_functions');
    if exist(functions_path, 'dir')
        addpath(genpath(functions_path));
        %logMessage(sprintf('Added functions path: %s', functions_path));
    else
        error(strcat('Functions path does not exist: ', string(functions_path)));
    end
    
    %% Calculate Frame Delta (dk)
    dk = dkRobust(c1n, c2n);
    logMessage(sprintf('Calculated frame delta (dk): %f', dk));
    
    %% Extract Calibration Trajectories
    calTraj = extractCalibrationTrajectories(df1, df2, dk);
    logMessage(sprintf('Total number of calibration trajectories: %d', size(calTraj.j1, 1)));
    
    % If calTraj big, randomly downsample for faster computation
    maxPoints = 10000;
    numPoints = size(calTraj.j1, 1);
    if numPoints > maxPoints
        idx = randperm(numPoints, maxPoints);
        calTraj.j1 = calTraj.j1(idx, :);
        calTraj.j2 = calTraj.j2(idx, :);
        logMessage(sprintf('Randomly sampled calibration trajectories from %d to %d points.', numPoints, maxPoints));
    end
    
    %% Load Camera Parameters
    paramsFile = 'sonya7r4--rokinon35mmf1.4.mat';
    if exist(paramsFile, 'file')
        S = load(paramsFile);  % Load into a structure
        if isfield(S, 'cameraParams')
            cameraParams = S.cameraParams;
            intrinsics = cameraParams.Intrinsics;
            K = intrinsics.K;
            %logMessage(sprintf('Loaded camera parameters from %s', paramsFile));
        else
            error('The file %s does not contain the variable cameraParams.', paramsFile);
        end
    else
        error('Camera parameters file not found: %s', paramsFile);
    end
    
    %% Calculation Method: "E" for Essential or "F" for Fundamental
    method = "E";  % Change to "F" if using Fundamental matrix method
    logMessage(sprintf('Calculation method set to: %s', method));
    
    stereoParams = struct();
    if method == "E"
        % Estimate Essential matrix, rotation and translation
        [E, R, t] = estimate_E_R_with_t_constraint(calTraj.j1(:,1:2), calTraj.j2(:,1:2), intrinsics);
        % Scale translation to real-world units
        t = t .* distance_between_camera_meters;
        stereoParams.t = -t;
        stereoParams.R = R;
        stereoParams.E = E;
        stereoParams.K = K;
        logMessage('Estimated E,R,t parameters using Essential matrix method.');
        
        % Matching points using the Essential method
        logMessage('Beginning point matching and triangulation with Essential matrix...');
        [matched_points_1, matched_points_2] = matchStereoWithE(df1, df2, stereoParams, dk, 10000);
        logMessage('Point matching complete.');
        
    elseif method == "F"
        % Normalize points and estimate Fundamental matrix
        points1 = calTraj.j1(:,1:2) ./ calTraj.j1(:,3);
        points2 = calTraj.j2(:,1:2) ./ calTraj.j2(:,3);
        logMessage('Estimating Fundamental matrix. This may take a while...');
        F = estimateFundamentalMatrix(points1, points2, 'Method', 'Norm8Point', ...
                                      'NumTrials', 10000, 'Confidence', 99.99, ...
                                      'DistanceThreshold', 1e-4);
        logMessage('Fundamental matrix estimation complete.');
        
        [E, R, t] = estimate_E_R_with_t_constraint(calTraj.j1(:,1:2), calTraj.j1(:,1:2), intrinsics);
        t = t .* distance_between_camera_meters;
        stereoParams.t = -t;
        stereoParams.R = R;
        stereoParams.E = E;
        stereoParams.F = F;
        stereoParams.K = K;
        logMessage('Estimated E,R,t, parameters using Fundamental matrix method.');
        
        % Matching points using the Fundamental method
        logMessage('Beginning point matching and triangulation with Fundamental matrix...');
        [matched_points_1, matched_points_2] = matchStereo(df1, df2, stereoParams, dk, 10000);
        logMessage('Point matching complete.');
    else
        error('Unknown calculation method: %s', method);
    end
    
    %% Triangulate 3D Points
    P1 = K * [eye(3), zeros(3,1)];
    P2 = K * [stereoParams.R, stereoParams.t];
    xyz = triangulate(matched_points_1(:,1:2), matched_points_2(:,1:2), P1', P2');
    logMessage('Triangulation complete.');
    
    % Append the timestamp to xyz
    xyzt = [xyz, matched_points_1(:,3)];
    
    % Reorient coordinates from camera (-x,z,y) to real (x,y,z)
    xyzt(:,1) = -xyzt(:,1);
    xyzt(:, [2,3]) = xyzt(:, [3,2]);
    %logMessage('Reoriented triangulated points to real-world coordinates.');
    
    %% Trajectorize and Post-Process Trajectories
    xyztkj = trajectorize(xyzt);
    
    % Optional post-processing scripts to set minimum number of points in a streak,
    % And the minimum number of streaks to make a trajectory

    % % Remove streaks with fewer than 4 detections
    % [unique_k, ~, k_idx] = unique(xyztkj(:,5));
    % streak_counts = accumarray(k_idx, 1);
    % valid_streaks = unique_k(streak_counts >= 4);
    % xyztkj = xyztkj(ismember(xyztkj(:,5), valid_streaks), :);
    % 
    % % Remove trajectories with fewer than 3 unique streaks
    % [unique_j, ~, j_idx] = unique(xyztkj(:,6));
    % unique_streaks_per_j = accumarray(j_idx, xyztkj(:,5), [], @(x) numel(unique(x)));
    % valid_trajectories = unique_j(unique_streaks_per_j >= 3);
    % xyztkj = xyztkj(ismember(xyztkj(:,6), valid_trajectories), :);
    % logMessage('Processed and filtered trajectories.');
    
    %% Optional: Plot 3D Reconstruction
    % figure;
    % scatter3(xyztkj(:,1), xyztkj(:,2), xyztkj(:,3), 20, xyztkj(:,6), 'filled');
    % xlabel('X'); ylabel('Y'); zlabel('Z');
    % title('3D Reconstruction');
    % logMessage('Displayed 3D reconstruction plot.');
    
    %% Build Output Structure
    result = struct();
    result.xyzt = xyzt;
    result.xyztkj = xyztkj;
    result.input_path = input_path;
    result.intrinsics = intrinsics;
    result.K = K;
    result.dk = dk;
    result.calTraj = calTraj;
    result.frame_width = frame_width;
    result.frame_height = frame_height;
    result.calculation_method = method;
    result.R = stereoParams.R;
    result.t = stereoParams.t;
    if isfield(stereoParams, 'E')
        result.E = stereoParams.E;
    end
    if isfield(stereoParams, 'F')
        result.F = stereoParams.F;
    end
    logMessage('Stereo reconstruction complete. Writing results.');

    %% Write struct to .mat file, and xyztkj to csv to <input foler>/../Sony_A7_xyzt/
    % To save the final struct (in the input data folder)
    % save(fullfile(input_path, 'result.mat'), 'result');
    
end



% Helper Function to Load xyt Files
function [df, n, ff] = loadXytFromFile(filePath)
    % loadXytFromFile Loads xyt data from a file.
    %   Supports .mat and .csv files.
    %
    %   [df, n, ff] = loadXytFromFile(filePath) returns the xyt data (df),
    %   frame numbers (n), and optionally the ff structure if available.
    %
    [~,~,ext] = fileparts(filePath);
    ff = [];  % default if not available
    if strcmpi(ext, '.mat')
        data = load(filePath);
        fns = fieldnames(data);
        if isempty(fns)
            error('The loaded .mat file %s does not contain expected fields.', filePath);
        end
        s = data.(fns{1});
        if isfield(s, 'xyt')
            df = s.xyt;
        else
            error('Field xyt not found in file: %s', filePath);
        end
        if isfield(s, 'n')
            n = s.n;
        else
            n = (1:size(df,1))';
        end
        if isfield(data, 'ff')
            ff = data.ff;
        end
    elseif strcmpi(ext, '.csv')
        df = readmatrix(filePath);
        if size(df,2) >= 3
            n = df(:,3);
        else
            n = (1:size(df,1))';
        end
    else
        error('Unsupported file extension: %s', ext);
    end
end

function logMessage(message)
% logMessage Simple logger that prints a message with a timestamp.
    timestamp = datetime('now');
    fprintf('[%s] %s\n', timestamp, message);
end
