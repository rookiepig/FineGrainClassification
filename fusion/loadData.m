%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: loadData.m
% Desc: load data for Group Fusion
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 'Load data ... \n' );

% load imadb, grp, kernel
global imdb;
global grp;

load( conf.imdbPath );
load( conf.grpPath );
load( conf.kernelPath );

% set clusterNum for each group system
GRP_SYS_NUM = numel( grp );


if( conf.useOracleCluster )
  % oracle cluter id for each test sample
  for g = 1 : GRP_SYS_NUM 
    grp{ g }.clusterNum = numel( grp{ g }.cluster );
  end
  for g = 1 : GRP_SYS_NUM
    grp{ g }.clsToCluster = zeros( size( imdb.clsLabel ) );
    for c = 1 : grp{ g }.clusterNum
      clsIdx = ismember( imdb.clsLabel, grp{ g }.cluster{ c } );
      grp{ g }.clsToCluster( clsIdx ) = c;
    end
  end
end

fprintf( '\n...Done\n' );
