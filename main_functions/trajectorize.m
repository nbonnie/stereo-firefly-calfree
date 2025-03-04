function xyztkj = trajectorize(xyzt, prm)
% TRAJECTORIZE processes xyzt data into streaks and trajectories.
%   xyztkj = trajectorize(xyzt, prm)
% 
%   Inputs:
%     xyzt - Nx4 matrix [x, y, z, t]
%     prm  - Parameter struct containing necessary linking parameters
%   Output:
%     xyztkj - Nx6 matrix [x, y, z, t, k (streak), j (trajectory)]

    if nargin < 2 || isempty(prm)
        % Pick these values carefully to fit your experiment
        prm.stk.linkRadiusMtr = 0.3; 
        prm.trj.linkRadiusMtr = 1.00;
        prm.trj.linkMinLagFrm = 0; 
        prm.trj.linkMaxLagFrm = 20;
    end

    disp(strcat(string(datetime('now')), ' -- Streak Creation started...'))
    
    % Convert xyzt to streaks
    [xyztk, strk] = xyzt2strk(xyzt, prm);

    disp(strcat(string(datetime('now')), ' -- Trajectorizing started...'))
    
    % Convert streaks to trajectories
    xyztkj = strk2traj(xyztk, strk, prm);

    disp(strcat(string(datetime('now')), ' -- Processing completed.'))
end

%% XYZT to streaks
function [xyztk, strk] = xyzt2strk(xyzt, prm)
% XYZT2STRK groups xyzt data into streaks based on spatial and temporal proximity.

    strkKernel = determineStrkKernel(xyzt);
    adj = buildSparseStrkAdj(xyzt, strkKernel, prm.stk.linkRadiusMtr);
    
    dg = digraph(adj);
    [strkID, strkDuration] = conncomp(dg, 'type', 'weak');
    
    xyztk = [xyzt, strkID(:)];
    xyztk = sortrows(xyztk, [5 4]);
    
    strk.xyzts = mat2cell(xyztk, strkDuration(:));
    strk.nStreaks = max(strkID);

    % streak first and last frames
    strk.ti = cellfun(@(x) x(1,4), strk.xyzts);
    strk.tf = cellfun(@(x) x(end,4), strk.xyzts);

    % streak first and last positions
    strk.ri = cell2mat(cellfun(@(x) x(1,1:3), strk.xyzts,'UniformOutput',false));
    strk.rf = cell2mat(cellfun(@(x) x(end,1:3), strk.xyzts,'UniformOutput',false));

end

function strkKernel = determineStrkKernel(xyzt)
    % simply the maximum number of flashes in a frame
    t = xyzt(:,4);
    n = histcounts(t,0.5:max(t)+0.5);
    strkKernel = max(n);
end


function adj = buildSparseStrkAdj(xyzt,strkKernel,strkLinkRadius)
    % number of adjacent streaks to probe for match
    x = xyzt(:,1);
    y = xyzt(:,2);
    z = xyzt(:,3);
    t = xyzt(:,4);
    p = length(t);
    
    % build sparse matrix
    sp = spdiags(ones(p,2*strkKernel+1),-strkKernel:strkKernel,p,p);
    [row,col] = find(sp);
    
    % flash delays
    dt = sparse(row, col, abs(t(row)-t(col)));
    
    % flash distances
    dx = sparse(row, col, x(row)-x(col)+eps); % +eps necessary to avoid numeric zero equated to sparse zero 
    dy = sparse(row, col, y(row)-y(col)+eps);
    dz = sparse(row, col, z(row)-z(col)+eps);
    dr = sqrt(dx.^2+dy.^2+dz.^2);
    
    
    % distance-based linkage (distance-adjacency matrix)
    adjt = (dt == 1);
    adjr = (spfun(@(S) S-strkLinkRadius,dr) < 0);
    adj = adjt & adjr;
end

%% Streaks to Trajectories
function xyztkj = strk2traj(xyztk, strk, prm)
% STRK2TRAJ links streaks into trajectories.
%    .trj.linkRadiusMtr : max distance between streaks (meters)
%    .trj.linkMinLagFrm : min delay between streaks (frames)
%    .trj.linkMaxLagFrm : max delay between streaks (frames)


    trajKernel = 100;
    adj = buildSparseTrajAdj(strk, trajKernel, prm.trj.linkRadiusMtr, prm.trj.linkMinLagFrm, prm.trj.linkMaxLagFrm);
    
    dg = digraph(adj);
    trajID = conncomp(dg, 'type', 'weak');
    
    trajID_expanded = trajID(xyztk(:,5)); % Use the streak IDs in xyztk to map trajectory IDs

    xyztkj = [xyztk, trajID_expanded(:)];
    %xyztkj(:,4) = xyztkj(:,4) ./ prm.mov.frameRate; % Convert time to seconds
end


function adj = buildSparseTrajAdj(strk,trajKernel,trajLinkRadius,trajLinkMinLagFrm,trajLinkMaxLagFrm)
    % builds sparse adjacency matrix (to avoid memory overload)
    
    % build sparse matrix
    nStreaks = strk.nStreaks; 
    sp = spdiags(ones(nStreaks,2*trajKernel+1),-trajKernel:trajKernel,nStreaks,nStreaks);
    [row,col] = find(sp);
    
    % flash delays
    dt = sparse(row, col, strk.ti(row)-strk.tf(col));
    dt = dt';
    
    
    % INVESTIGATE: DISTANCES ARE BEING CONNECTED WHEN THEY ARE TOO LARGE
    % flash distances
    dx = sparse(row, col, strk.ri(row,1)-strk.rf(col,1));
    dy = sparse(row, col, strk.ri(row,2)-strk.rf(col,2));
    dz = sparse(row, col, strk.ri(row,3)-strk.rf(col,3));
    dr = sqrt(dx.^2+dy.^2+dz.^2);
    dr = dr';
    
    
    % distance-based linkage (distance-adjacency matrix)
    adjtm = dt > trajLinkMinLagFrm;
    adjtM = spfun(@(S) S-trajLinkMaxLagFrm,dt) < 0;
    adjrM = spfun(@(S) S-trajLinkRadius,dr) < 0;
    adj = adjtm & adjtM & adjrM;
end