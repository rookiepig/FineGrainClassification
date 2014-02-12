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

w0 = cell( 1, nCluster );

clsToCluster = zeros( nClass, nCluster );
for t = 1 : nCluster
  PrintTab();fprintf( '  cluster %d\n', t );
  grpCls = curGrp.cluster{ t };

  % combine feature
  trainScore{ t } = svmScore( train, grpCls );
  allScore{ t } = svmScore( :, grpCls );
  % set classs to cluster matrix
  clsToCluster( grpCls, t ) = 1;

  % independent feature
  curTrain = intersect( find( ismember( imdb.clsLabel, grpCls ) ), train );
  curScore = svmScore( curTrain, grpCls );
  % map class label sequentially
  tmpLabel = imdb.clsLabel( curTrain );
  curLab = zeros( size( tmpLabel ) );
  for c = 1 : length( grpCls )
    curLab( tmpLabel == grpCls( c ) ) = c;
  end
  % softmax L2 regression get independent weight
  curAllScore = svmScore( :, grpCls );
  [ w0{ t }, proAll ] = MultiLRL2( curScore, curLab, ...
    curAllScore, 1, ones( length( curLab ), 1 ) );
end % end for each cluster


[ wSoftmax, probFeat ] = MultiLRL2_all( trainScore, trainLab, clusterProb( train, : ), ...
  allScore, clusterProb, clsToCluster, 1, w0 );

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainMapSoft
