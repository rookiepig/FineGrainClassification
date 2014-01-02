%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: prepareGroup.m
% Desc: init grp data structure using spectral clustering
% Author: Zhang Kang
% Date: 2014/01/02 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% grp Data:
% - grp.clusterNum - 
% - grp.cluster      - 1 * clusterNum cell
% - grp.clsToCluster - nSample * 1 indicator

% get 
GRP_SYS_NUM = 7;
grp = cell( 1, GRP_SYS_NUM );
k = 2;
for g = 1 : GRP_SYS_NUM
  grp{ g }.clusterNum = k;

  k = k * 2;
end

