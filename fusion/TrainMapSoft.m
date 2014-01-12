function [ probFeat ] = TrainMapSoft( conf, imdb, curGrp, svmScore )
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

for t = 1 : nCluster
  PrintTab();fprintf( '\t cluster %d\n', t );
  grpCls = curGrp.cluster{ t };
  % train feature
  curTrain = intersect( find( ismember( imdb.clsLabel, grpCls ) ), train );
  trainScore = svmScore( curTrain, grpCls );
  % map class label sequentially
  tmpLabel = imdb.clsLabel( curTrain );
  trainLabel = zeros( size( tmpLabel ) );
  for c = 1 : length( grpCls )
    trainLabel( tmpLabel == grpCls( c ) ) = c;
  end
  % softmax L2 regression
  allScore = svmScore( :, grpCls );
  [ ~, proAll ] = MultiLRL2( trainScore, trainLabel, allScore );
  % set final probability
  probFeat( :, grpCls ) = proAll;
end % end for each cluster

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainMapSoft
