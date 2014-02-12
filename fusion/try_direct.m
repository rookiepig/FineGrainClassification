% direct combine cluster probability and class probability

% init basic variables
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
nCluster = curGrp.nCluster;
train   = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );
test    = find( imdb.ttSplit == 0 );
testLab  = imdb.clsLabel( test );

probFeat = zeros( nSample, nClass );

for t = 1 : nCluster
  fprintf( '  cluster %d\n', t );
  grpCls = curGrp.cluster{ t };

  % get cluster prior prob
  clusterProb = curGrp.clusterProb( :, t );

  probAll = curModel.bayesProb( :, grpCls );
  Z = max( sum( sqrt( probAll .* probAll ), 2 ), eps );
  probAll = bsxfun( @rdivide, probAll, Z );

  % bayes combine
  probFeat( :, grpCls ) = probFeat( :, grpCls ) + ...
    bsxfun( @times, probAll, clusterProb );
end % end for each cluster


% get train and test accuracy

[ ~, trainAcc ] = ScoreToConf( probFeat( train, : ), trainLab );
fprintf( 'Train Acc: %.2f %%\n', trainAcc );

[ ~, testAcc ]  = ScoreToConf( probFeat( test, : ), testLab );
fprintf( 'Test Acc: %.2f %%\n', testAcc );

[ ~, testAcc ]  = ScoreToConf( curModel.bayesProb( test, : ), testLab );
fprintf( 'Test Acc: %.2f %%\n', testAcc );


[ ~, testAcc ]  = ScoreToConf( curModel.bayesProb( test, : ) + probFeat( test, : ), testLab );
fprintf( 'Test Acc: %.2f %%\n', testAcc );

