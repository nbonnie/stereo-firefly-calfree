clear;clc;clf;

%% Read in files
folder_path = "PATH";

addpath(genpath(folder_path))
fnames = dir(strcat(folder_path,"/*.mat"));
data1 = load(strcat(folder_path, "/",fnames(1).name));
data2 = load(strcat(folder_path, "/",fnames(2).name));


%% Loads in corresponding data, extracts information needed
fns1 = fieldnames(data1);
fns2 = fieldnames(data2);

df1 = data1.(fns1{1}).xyt;
df2 = data2.(fns2{1}).xyt;
c1n = data1.(fns1{1}).n;
c2n = data2.(fns2{1}).n;

try
    frame_width = size(data1.ff.bkgrStack,2);
    frame_height = size(data1.ff.bkgrStack,1);
catch
    frame_width = 1920;
    frame_height = 1080;
    disp("Warning: Movie information not read in from files")
end

max_y = max([df1(:,2); df2(:,2)]);

% Flip the y-values
df1(:,2) = max_y - df1(:,2);
df2(:,2) = max_y - df2(:,2);

%% This is where you would remove persistent objects
% Some simple time splits based for 06/02 data
df1(df1(:,3) <= 8000, :) = [];
df2(df2(:,3) <= 8000, :) = [];

df1(df1(:,2) > 680, :) = [];
df2(df2(:,2) > 670, :) = [];

[~,~,ix1] = unique(df1(:,3));
C1 = accumarray(ix1,1).';
toRemove = C1(ix1) > 10; % This finds bright detections like headlamps or other spontaneous lights
df1 = df1(~toRemove, :);


[~,~,ix2] = unique(df2(:,3));
C2 = accumarray(ix2,1).';
toRemove = C2(ix2) > 10;
df2 = df2(~toRemove, :);


%% Load in functions
addpath(genpath("/Users/nbonnie/Desktop/stereo-firefly-calfree/main_functions"))

%% Calculates the time difference in frames
dk = dkRobust(c1n, c2n);
disp(strcat("Frame delta: ",string(dk)))
fprintf("\n\n")

%% Find all 1-flash-frames to calibrate on:
calTraj = extractCalibrationTrajectories(df1,df2,dk);

% Set maximum number of points to use
maxPoints = 8000;

% Get the total number of points
numPoints = size(calTraj.j1, 1);

% If we have more than maxPoints, randomly sample (increase iterations
% will have diminishing returns past ~3000 points (at the cost of exponential computation)
% This maxPoints value could be determined in a more robust way in the future
if numPoints > maxPoints
    idx = randperm(numPoints, maxPoints); % Randomly select indices
    calTraj.j1 = calTraj.j1(idx, :);
    calTraj.j2 = calTraj.j2(idx, :);
end

%% Pull in camera intrinsic matrix and create CameraIntrinsics variable
% load('sony_camera_parameters.mat')
% K = sony_camera_parameters.CameraParameters1.K; % Theoretically this should be the same for every camera-lens pair
% intrinsics = K_2_cameraIntrinsics(K, 1920, 1080);

load('sonya7r4--rokinon35mmf1.4.mat')
intrinsics = cameraParams.Intrinsics;
K = intrinsics.K;

method = "E"; % E for Essential calculation, F for Fundamental calculation

if method == "E"
    % Estimate R, t (with constraint), and E
    [E, R, t]= estimate_E_R_with_t_constraint(calTraj.j1(:,1:2), calTraj.j2(:,1:2), intrinsics);
    distance_between_camera_meters = 0.9906; % Tune to each experiment measurement
    
    % Scale traslation matrix to use real world units
    t = t ./ (1/distance_between_camera_meters);
    
    % Store parameters for matching
    stereoParams.t = -t;
    stereoParams.R = R;
    stereoParams.E = E;
    stereoParams.K = intrinsics.K;

    % Match Points with E
    disp('Beginning point matching and triangulation...')
    disp(datetime('now'))
    % Finds matching points frame by frame after solving epipolar constraint
    [matched_points_1,matched_points_2]=matchStereoWithE(df1, df2, stereoParams, dk, 10000);
    disp('Matching complete.')
    disp(datetime('now'))

elseif method == "F"
    % Calculate F matrix
    % normalize to get third coordinate equal to 1 (see Matlab doc)
    points1 = calTraj.j1(:,1:2)./calTraj.j1(:,3);
    points2 = calTraj.j2(:,1:2)./calTraj.j2(:,3);
    
    %calculate fundamental matrix
    disp('Estimating fundamental matrix; this may take a while (up to ~1hr)...')
    disp(datetime('now'))
    F  = estimateFundamentalMatrix(points1, points2, 'Method', 'Norm8Point', ...
                                                    NumTrials=10000, ...
                                                    Confidence=99.99, ...
                                                    DistanceThreshold=1e-4);
    disp('calibration complete.')
    disp(datetime('now'))

    % Estimate E, R, t with t constraint
    [E, R, t]= estimate_E_R_with_t_constraint(calTraj.j1(:,1:2), calTraj.j1(:,1:2), intrinsics);
    distance_between_camera_meters = 0.9906; % Tune to each experiment measurement
    
    % Scale traslation matrix to use real world units
    t = t ./ (1/distance_between_camera_meters);

    % Store parameters for matching
    stereoParams.t = -t;
    stereoParams.R = R;
    stereoParams.E = E;
    stereoParams.F = F;
    stereoParams.K = K;

    % Match Points with F
    disp('Beginning point matching and triangulation...')
    disp(datetime('now'))
    % Finds matching points frame by frame after solving epipolar constraint
    [matched_points_1,matched_points_2]= matchStereo(df1, df2, stereoParams, dk, 10000);
    disp('Matching complete.')
    disp(datetime('now'))
end



%% Triangulate
P1 = K * [eye(3), zeros(3,1)];
P2 = K * [stereoParams.R, stereoParams.t];
xyz = triangulate(matched_points_1(:,1:2), matched_points_2(:,1:2), P1', P2');
disp('Triangulation complete.')
disp(datetime('now'))
xyzt = [xyz, matched_points_1(:,3)];

% Flip back from camera coordinates (-x,z,y) to real coordinates (x,y,z)
xyzt(:,1) = -xyzt(:,1);
xyzt(:, [2, 3]) = xyzt(:, [3, 2]);

%% Trajectorize
xyztkj = trajectorize(xyzt);

% Remove streaks with fewer than 2 points
[unique_k, ~, k_idx] = unique(xyztkj(:,5));  % Unique streak IDs
streak_counts = accumarray(k_idx, 1);  % Cout occurrences of each streak ID
valid_streaks = unique_k(streak_counts >= 4);  % Keep streaks with 2 or more points
xyztkj = xyztkj(ismember(xyztkj(:,5), valid_streaks), :);  % Filter data

% Remove trajectories with fewer than 3 unique streaks
[unique_j, ~, j_idx] = unique(xyztkj(:,6));  % Unique trajectory IDs
unique_streaks_per_j = accumarray(j_idx, xyztkj(:,5), [], @(x) numel(unique(x)));  % Count unique streaks per trajectory
valid_trajectories = unique_j(unique_streaks_per_j >= 3);  % Keep trajectories with 3 or more unique streaks
xyztkj = xyztkj(ismember(xyztkj(:,6), valid_trajectories), :);  % Filter data

%% Plot
figure;
scatter3( xyztkj(:,1) , xyztkj(:,2) , xyztkj(:,3), 20, xyztkj(:,6), 'filled')
xlabel('X'); ylabel('Y'); zlabel('Z');