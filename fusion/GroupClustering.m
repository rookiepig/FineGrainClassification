function [ oneGrp ] = GroupClustering( nCluster )
%% GroupClustering
%  Desc: clustering classes for one group
%  In: 
%    nCluster -- number of clusters
%  Out:
%    oneGrp -- (struct) 
%      - oneGrp.nCluster     - number of clusters
%      - oneGrp.cluster      - 1 * nCluster cell
%      - oneGrp.clsToCluster - nSample * 1 indicator
%%

PrintTab();fprintf( 'function: %s\n', mfilename );

% get configuration
conf = InitConf( );
% load imdb, kernel
load( conf.imdbPath );

% get oneGrp struct
oneGrp.nCluster = nCluster;
oneGrp.cluster = cell( 1, nCluster );
% clusterring
PrintTab();fprintf( '\t Cluaster method: %s\n', conf.clusterType );
PrintTab();fprintf( '\t Cluaster number: %d\n', nCluster );
switch conf.clusterType
  case 'spectral'
    % Spectral clustering
    PrintTab();fprintf( '\t loading kernel (maybe slow)\n' );
    load( conf.kernelPath );
    % get similarity matrix
    clsSim = KernelToSim( kernel, imdb.clsLabel, imdb.ttSplit );
    [ C, ~, ~ ] = SpectralClustering( clsSim, nCluster, 3 );
    for  k = 1 : nCluster
      oneGrp.cluster{ k } = find( C == k );
    end
  case 'tree'
    % load from phylogeny tree
    PrintTab();fprintf( '\t Load phylogeny tree manually\n' );
  case 'confusion'
    % use confusion matrix of training data as clsSim
    load( 'clsSim.mat' );
    [ C, ~, ~ ] = SpectralClustering( clsSim, nCluster, 3 );
    for  k = 1 : nCluster
      oneGrp.cluster{ k } = find( C == k );
    end
  otherwise
    PrintTab();
    fprintf( '\t Error: unknow clustering method %s\n', conf.clusterType );
end

% end function GroupClustering