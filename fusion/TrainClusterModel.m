function [ curGrp ] = TrainClusterModel( curGrp )
%% TrainClusterModel
%  Desc: train cluster model --> get test cluster label
%  In: 
%    curGrp -- (struct) clustering infomation for one group
%  Out:
%    curGrp -- (struct) 
%      - curGrp.nCluster  - number of clusters
%      - curGrp.cluster      - (1 * nCluster) class index
%      - curGrp.clusterLabel - (nSample * 1) ground truth cluster label 
%      - curGrp.clusterSVM   - (1 * nCluster) SVM model
%      - curGrp.clusterScore - (nSample * cluterNum) SVM score
%      - curGrp.clusterConf  - (nCluster * nCluster) confusion matrix
%      - curGrp.clsToCluster - (nSample * 1) cluster for each sample
%%

fprintf( 'function: %s\n', mfilename );
tic;

% get configuration
conf = InitConf( );
% load imdb, kernel
load( conf.imdbPath );
fprintf( '\t loading kernel (maybe slow)\n' );
load( conf.kernelPath );

% init training and testing kernel
train = find( imdb.ttSplit == 1 );
test  = find( imdb.ttSplit == 0 );
trainK = kernel( train, train );
trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
testK  = kernel( test, train );
testK = [ ( 1 : size( testK, 1 ) )', testK ];

% get cluster label
curGrp.clusterLabel = zeros( size( imdb.clsLabel ) );
for k = 1 : curGrp.nCluster
  clusterIdx = find( ismember( imdb.clsLabel, curGrp.cluster{ k } ) );
  curGrp.clusterLabel( clusterIdx ) = k;
end
% train and test cluster SVM
clsNum = curGrp.nCluster;
curGrp.clusterSVM   = cell( 1, clsNum );
curGrp.clusterScore = cell( 1, clsNum );
for c = 1 : clsNum
  fprintf( '\t cluster train test: %d (%.2f %%)\n', c, 100 * c / clsNum );
  y = 2 * ( curGrp.clusterLabel == c ) - 1;
  % train
  curGrp.clusterSVM{ c } = libsvmtrain( double( y( train ) ), ...
    double( trainK ), conf.clusterSVMOPT );
  % test
  [ ~,~,curGrp.clusterScore{ c } ] = libsvmpredict( double( y( test ) ), ...
    double( testK ), curGrp.clusterSVM{ c } );
end
% set train and test class to cluster label
curGrp.clusterScore = cat( 2, curGrp.clusterScore{ : } );
[ ~, testPred ] = max( curGrp.clusterScore, [], 2 );
curGrp.clsToCluster = zeros( size( imdb.clsLabel ) );
curGrp.clsToCluster( train ) = curGrp.clusterLabel( train );
curGrp.clsToCluster( test ) = testPred;

% get confusion matrix
confusion = confusionmat( curGrp.clusterLabel( test ), testPred );
for c = 1 : clsNum
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) / sumC;
end
fprintf( '\t mean accuracy: %.2f %%\n', 100 * mean(diag(confusion)) );
curGrp.clusterConf = confusion;

fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainClusterModel