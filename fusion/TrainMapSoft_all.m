function [ wSoftmax, probFeat ] = TrainMapSoft_all( conf, imdb, curGrp, svmScore )
%% TrainMapSoft
%  Desc: softmax regression to map each cluster SVM probFeat to prob
%  In: 
%    conf, imdb -- basic variables
%    curGrp -- (struct) group cluster info
%    svmScore -- (nSample * nClass) SVM feature with test SVM feat
%  Out:
%    probFeat  -- (nSample * nClass) probability output for each cluster
%%

PrintTab();fprintf( 'function: %s\n', mfilename );
tic;

% init basic variables
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
nCluster = curGrp.nCluster;
train   = find( imdb.ttSplit == 1 );
probFeat = zeros( nSample, nClass );

% training label
trainLab = imdb.clsLabel( train );
clusterProb = curGrp.clusterProb;

clsToCluster = zeros( nClass, nCluster );
for t = 1 : nCluster
  PrintTab();fprintf( '  cluster %d\n', t );
  grpCls = curGrp.cluster{ t };

  % set classs to cluster matrix
  clsToCluster( grpCls, t ) = 1;

  % train feature
  trainScore{ t } = svmScore( train, grpCls );
  % softmax L2 regression
  allScore{ t } = svmScore( :, grpCls );
  
  % set final probability
  % clusterProb = repmat( clusterProb, [ 1 length( grpCls ) ] );
  % bayes combine
  % probFeat( :, grpCls ) = probFeat( :, grpCls ) + proAll .* clusterProb;
end % end for each cluster


[ wSoftmax, probFeat ] = MultiLRL2_all( trainScore, trainLab, clusterProb( train, : ), ...
  allScore, clusterProb, clsToCluster, 1 );

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainMapSoft
