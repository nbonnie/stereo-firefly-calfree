function [matched_df1, matched_df2] = matchStereoWithE(df1, df2, stereoParams, dk, thresh)
% MATCHSTEREOWITHE  Matches corresponding points between two stereo images using the essential matrix.
%
% Description:
%   This function finds corresponding feature points between two images captured by a 
%   calibrated stereo camera system. It uses the **essential matrix (E)** to enforce 
%   the epipolar constraint, ensuring that matched points lie on their respective 
%   epipolar lines. The function also accounts for any time offsets between the two 
%   cameras (dk) and applies a similarity threshold to refine the matches.
%
% Inputs:
%   df1:          (nx3) Matrix of detected features from camera 1. Columns represent (x, y, time).
%   df2:          (mx3) Matrix of detected features from camera 2. Columns represent (x, y, time).
%   stereoParams: Struct containing stereo calibration parameters, including:
%                 - E:  Essential matrix
%                 - K:  Intrinsic matrix for cameras 1 & 2
%   dk:           Scalar time offset (synchronization difference) between cameras.
%   thresh:       Distance threshold for epipolar constraint matching (default: 100 pixels). 
%
% Outputs:
%   matched_df1:  (kx3) Matrix of matched features from camera 1 (x, y, time).
%   matched_df2:  (kx3) Matrix of matched features from camera 2 (x, y, time).
%                 Each row in 'matched_df1' corresponds to the same feature as the 
%                 same row in 'matched_df2'.  
%
% Example Usage:
%   [matched_points1, matched_points2] = matchStereoWithE(features1, features2, stereoParams, dk); 
%
% Nolan R Bonnie, 03/2024, Updated 02/2025
% nolan.bonnie@colorado.edu

E = stereoParams.E;
K = stereoParams.K;

if nargin == 4
    thresh = 100;
end

% Sync times
df2(:,3) = df2(:,3)-dk;

% Init vars for loop
matched_df1 = [];
matched_df2 = [];

for t = intersect( unique(df1(:,3)) , unique(df2(:,3)) )'
    % Time will be trimmed out in point_matching_model()
    df1_t = df1(df1(:,3)==t, :);
    df2_t = df2(df2(:,3)==t, :);

    % Perform epipolar matching using the essential matrix
    [matched_df1_t, matched_df2_t, ~] = epipolarConstraintWithE(df1_t, df2_t, E, K, thresh);
    
    % Append new matched points at time t into complete lists
    % Possible to pre-allocate but likely not neccesary
    matched_df1 = [matched_df1; matched_df1_t]; %#ok<AGROW> 
    matched_df2 = [matched_df2; matched_df2_t]; %#ok<AGROW> 

end

end

function [matched_df1, matched_df2, distances] = epipolarConstraintWithE(df1, df2, E, K, thresh)
% EPIPOLAR_CONSTRAINT  Identifies potential feature matches using the essential matrix.
%
% Description:
%   Calculates distances of feature points in 'df2' to their corresponding epipolar 
%   lines defined by 'df1' and the essential matrix (E). Applies a distance threshold 
%   to filter potential matches.
%
% Inputs:
%   df1:          (nx3) matrix of features from camera 1 (x, y, time).
%   df2:          (mx3) matrix of features from camera 2 (x, y, time).
%   E:            Essential matrix.
%   K:            (3x3) Intrinsic matrix for camera 1 (& camera 2)
%   thresh:       Maximum distance for a feature to be considered a potential match.
%
% Outputs:
%   matched_df1:  (kx3) Matches from camera 1 (x, y, time).
%   matched_df2:  (kx3) Matches from camera 2 (x, y, time).
%   distances:    (mxn) Matrix of distances between points in 'df2' and epipolar lines.

% Convert pixel coordinates to normalized image coordinates (only x, y)
df1_norm_xy = (K \ [df1(:, 1:2), ones(size(df1, 1), 1)]')';
df2_norm_xy = (K \ [df2(:, 1:2), ones(size(df2, 1), 1)]')';

% Attach time column without modification
df1_norm = [df1_norm_xy(:, 1:2), df1(:, 3)];
df2_norm = [df2_norm_xy(:, 1:2), df2(:, 3)];

% Compute distances using the essential matrix
distances = dist2line(df1_norm, df2_norm, E);

% Find matches based on distance threshold
[matched_norm1, matched_norm2] = findMatches(df1_norm, df2_norm, distances, thresh);

% Convert matched points back to pixel coordinates
if ~isempty(matched_norm1)
    % Reapply intrinsic matrix to return to pixel space (only x, y)
    matched_df1_homog = (K * [matched_norm1(:, 1:2), ones(size(matched_norm1, 1), 1)]')';
    matched_df2_homog = (K * [matched_norm2(:, 1:2), ones(size(matched_norm2, 1), 1)]')';
    
    % Normalize homogeneous coordinates back to (x, y)
    matched_df1_xy = matched_df1_homog(:, 1:2) ./ matched_df1_homog(:, 3);
    matched_df2_xy = matched_df2_homog(:, 1:2) ./ matched_df2_homog(:, 3);
    
    % Restore time column from original normalized matches
    matched_df1 = [matched_df1_xy, matched_norm1(:, 3)];
    matched_df2 = [matched_df2_xy, matched_norm2(:, 3)];
else
    matched_df1 = [];
    matched_df2 = [];
end

end

function [matched_df1,matched_df2] = findMatches(df1,df2,distances,threshold)
% FINDMATCHES  Confirms feature matches based on the epipolar constraint and distance.
%
% Description:
%   Applies the Hungarian (Munkres) algorithm to assign feature points in 'df2' to their 
%   closest epipolar lines (within a distance threshold) defined by features in 'df1'.
%
% Inputs:
%   df1:          (nx3) matrix of features from camera 1 (x, y, time).
%   df2:          (mx3) matrix of features from camera 2 (x, y, time).
%   distances:    (mxn) Matrix of point-to-epipolar-line distances. 
%   threshold:    Maximum distance for a feature match to be valid.
%
% Outputs:
%   matched_df1:  (kx3) Confirmed matches from camera 1.
%   matched_df2:  (kx3) Confirmed matches from camera 2.

% Hungarian (Munkres) algorithm finds optimal matches between points and epipolar lines.
% 'M' will contain pairs of indices representing matches.
M = matchpairs(distances,threshold);

% Handle cases where no matches are found:
if isempty(M)
    matched_df1 = [];
    matched_df2 = [];
    return
end

% Extract matched point pairs
matched_df1 = zeros(size(M,1), 3);
matched_df2 = zeros(size(M,1), 3);
for i = 1:size(M, 1)
    % Slices out matched points derrived from the Hungarian algorithm
    % M(i, 1) is the index of a point in df2
    % M(i, 2) is the index of its corresponding matched point in df1
    matched_df1(i,:) = df1(M(i,2),:);
    matched_df2(i,:) = df2(M(i,1),:);
end

end

function distances = dist2line(df1, df2, E)
%DIST2LINE Calculates distances between points and epipolar lines in normalized coordinates.
%
% Inputs:
%   df1:          (nx3) matrix of features in normalized coordinates from camera 1.
%   df2:          (mx3) matrix of features in normalized coordinates from camera 2.
%   E:            Essential matrix.
%
% Outputs:
%   distances:    (mxn) Matrix of distances between points in 'df2' and their epipolar lines.    

abc = df1 * E; % Compute epipolar lines in camera 2
distances = abs( (abc(:,1)'.*df2(:,1)) + (abc(:,2)'.*df2(:,2)) + abc(:,3)' ) ./ sqrt( (abc(:,1)').^2 + (abc(:,2)').^2 );

end