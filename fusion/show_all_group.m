%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: show_all_group.m
% Desc: show all groups information
% Author: Zhang Kang
% Date: 204/01/04
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% get configuration
conf = InitConf( );

% cache file name
cacheGrpInfo = cell( 1, conf.nGroup );
for g = 1 : conf.nGroup
  fName = sprintf( '-tmpGrpInfo%03d.mat',  g );
  cacheGrpInfo{ g } = fullfile( conf.cacheDir, [ conf.prefix, fName ] );
end
cacheGrpModel = cell( 1, conf.nGroup );
for g = 1 : conf.nGroup
  fName = sprintf( '-tmpGrpModel%03d.mat',  g );
  cacheGrpModel{ g } = fullfile( conf.cacheDir, [ conf.prefix, fName ] );
end


clusterResult = cell( 1, conf.nGroup );
clsResult = cell( 1, conf.nGroup );
mapResult = cell( 1, conf.nGroup );
for g = 1 : conf.nGroup
  load( cacheGrpInfo{ g } );
  load( cacheGrpModel{ g } );
  [ clusterResult{ g }, clsResult{ g }, mapResult{ g } ] = ...
    ShowOneGroup( conf, curGrp, curModel );
end
