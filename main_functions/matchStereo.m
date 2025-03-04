function [matched_df1, matched_df2] = matchStereo(df1, df2, stereoParams, dk, thresh)
% MATCHSTEREO Finds corresponding points between two stereo images.
%
% Description:
%   This function establishes point correspondences between two sets of image 
%   features (df1 and df2) from a calibrated stereo camera setup. It leverages 
%   the epipolar constraint to find potential matches and enforces a similarity
%   threshold to refine the results.
%
% Inputs:
%   df1:          (nx3) matrix of features from camera 1. Columns represent x, y, and time.
%   df2:          (mx3) matrix of features from camera 2. Columns represent x, y, and time.
%   stereoParams: Struct containing stereo calibration parameters, including the fundamental matrix (F).
%   dk:           Time offset (delay/synchronization difference) between cameras.
%   thresh:       Similarity threshold for matching (optional, default is 100). 
%
% Outputs:
%   matched_df1:  (kx3) matrix of matched features from camera 1 (x, y, and time).
%   matched_df2:  (kx3) matrix of matched features from camera 2 (x, y, and time).
%                 Each row in 'matched_df1' corresponds to the same feature as the 
%                 same row in 'matched_df2'.     
%
% Example Usage:
%   [matched_points1, matched_points2] = matchStereo(features1, features2, stereoParams, dk); 
%
% Nolan R Bonnie, 03/2024
% nolan.bonnie@colorado.edu

if nargin == 4
    thresh = 100;
end

% Sync times
df2(:,3) = df2(:,3)-dk;

% Init vars for loop
matched_df1 = [];
matched_df2 = [];

F = stereoParams.F;

for t = intersect( unique(df1(:,3)) , unique(df2(:,3)) )'
    % Time will be trimmed out in point_matching_model()
    df1_t = df1(df1(:,3)==t, :);
    df2_t = df2(df2(:,3)==t, :);

    [matched_df1_t, matched_df2_t, ~] = epipolar_constraint(df1_t, df2_t, F, thresh);
    
    % Append new matched points at time t into complete lists
    % Possible to pre-allocate but likely not neccesary
    matched_df1 = [matched_df1; matched_df1_t]; %#ok<AGROW> 
    matched_df2 = [matched_df2; matched_df2_t]; %#ok<AGROW> 

end

end


function [matched_df1, matched_df2, distances] = epipolar_constraint(df1, df2, F, thresh)
% EPIPOLAR_CONSTRAINT  Identifies potential feature matches using unbalanced epipolar constraint.
%
% Description:
%   Calculates distances of feature points in 'df2' to their corresponding epipolar 
%   lines defined by 'df1' and the fundamental matrix (F). Applies a distance threshold 
%   to filter potential matches.
%
% Inputs:
%   df1:          (nx3) matrix of features from camera 1  (x, y, time).
%   df2:          (mx3) matrix of features from camera 2  (x, y, time).
%   F:            Fundamental matrix from stereoParams.
%   thresh:       Maximum distance for a feature to be considered a potential match. 
%
% Outputs:
%   matched_df1:  (kx3) Matches from camera 1 (x, y, time).
%   matched_df2:  (kx3) Matches from camera 2 (x, y, time).
%   distances:    (mxn) Matrix of distances between points in 'df2' and epipolar lines.
%
% Nolan R Bonnie, 03/2024
% nolan.bonnie@colorado.edu

% The Epipolar Constraint Formula:
%   [x1, y1, 1] * F * [x2; y2; 1] = 0
%
% Where:
%   * (x1, y1): 2D coordinates of a point in the first image
%   * (x2, y2): 2D coordinates of the corresponding point in the second image
%   * F: The fundamental matrix relating the two camera views
%
% Purpose:
%   Utilize numerical optimization to approximate which points from df2 @ time t
%   are closest to the lines produced by [x1, y1, 1] * F. If we apply this
%   algorithm to all of the lines and all of the points, we can then use
%   the assignment problem to locate which points match with which lines to
%   minimize the distances between all matches.

% Create copies of df1, df2 without time, and with 1s in (:,3) spot
df1_ec = df1(:, 1:2);
df2_ec = df2(:, 1:2);

% Setting up matrices to fit epipolar constraint equation
df1_ec(:,3) = 1;
df2_ec(:,3) = 1;

% Calculate the distances between all lines and points
distances = dist2line(df1_ec, df2_ec, F);

% Send in df1 and df2 (instead of the epipolar constraint (ec)
% counterparts) to slice out matching points that include time 
[matched_df1,matched_df2] = findMatches(df1, df2, distances, thresh);
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

function distances = dist2line(df1,df2,F)
%DIST2LINE Calculates distances between points and epipolar lines.
%
% Description:
%   Calculates the perpendicular distances between points in 'df2' and their 
%   corresponding epipolar lines defined by the fundamental matrix (F) and points in 'df1'.
%   Solves the whole problem in matrix form for efficiency.
%
% Inputs:
%   df1:          (nx3) matrix of features from camera 1 (x, y, time).
%   df2:          (mx3) matrix of features from camera 2 (x, y, time).
%   F:            Fundamental matrix from stereoParams
%
% Outputs:
%   distances:    (mxn) Matrix of distances between points in 'df2' and their epipolar lines.    
abc = df1*F; % ax+by+c = 0. Solution is in matrix form
distances = abs( (abc(:,1)'.*df2(:,1)) + (abc(:,2)'.*df2(:,2)) + abc(:,3)' ) ./ sqrt( (abc(:,1)').^2 + (abc(:,2)').^2 );
% distances matrix will be a matrix of size(df2,1) x size(df1,2)
% Each col represents the distances between the nth line (df1(n)*F) and
% each point in df2
end

function y = fund(v, F, x) %#ok<DEFNU> 
%FUND Calculates the y-coordinate of a point on an epipolar line.
%
% Description:
%   Given a vector 'v', the fundamental matrix (F), and an x-coordinate 'x', this function 
%   calculates the corresponding y-coordinate on the epipolar line defined by 'v*F'.
%
% Inputs:
%   v:      (1x3) Feature point from camera 1, of form [x,y,t].
%   F:      Fundamental matrix from stereoParams.
%   x:      x-coordinate of a point on the epipolar line. Should be
%   linspace object.
%
% Outputs:
%   y:      y-coordinate of the point on the epipolar line.
abc = v*F; % ax+by+c = 0
a = abc(1);
b = abc(2);
c = abc(3);
y = ((-x.*a) - (c)) / (b);
end