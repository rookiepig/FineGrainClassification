% try simple softmax
% x = x - max(x)

conf = InitConf();
load( conf.imdbPath );
load( conf.grpModelPath );
load( conf.grpInfoPath );


test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );

nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );

scores = zeros( nSample, nClass );

for g = 1 : conf.nGroup
  fprintf( 'group %d\n', g );
  curGrp = grpInfo{ g };
  curModel = grpModel{ g };
  clusterGtLab = curGrp.clusterGtLab;
  [ ~, trainAcc ] = ScoreToConf( curGrp.clusterScore( train, : ), ...
    clusterGtLab( train ) );
  fprintf( '\t cluster score train acc -- %.2f %%\n', trainAcc );
  [ ~, testAcc ] = ScoreToConf( curGrp.clusterScore( test, : ), ...
    clusterGtLab( test ) );
  fprintf( '\t cluster score test acc -- %.2f %%\n', testAcc );
  % simple softmax --> convert to probability
  curGrp.clusterProb = SimpleSoftMax( curGrp.clusterScore );

  [ ~, trainAcc ] = ScoreToConf( curGrp.clusterProb( train, : ), ...
    clusterGtLab( train ) );
  fprintf( '\t cluster prob train acc -- %.2f %%\n', trainAcc );
  [ ~, testAcc ] = ScoreToConf( curGrp.clusterProb( test, : ), ...
    clusterGtLab( test ) );
  fprintf( '\t cluster prob test acc -- %.2f %%\n', testAcc );

  nCluster = curGrp.nCluster;
  % simple softmax --> class probability
  for t = 1 : nCluster
    % PrintTab();fprintf( '\t\t cluster %d\n', t );
    grpCls = curGrp.cluster{ t };
    % softmax L2 regression
    allScore = curModel.svmScore( :, grpCls );
    probAll = SimpleSoftMax( allScore );
    curModel.probFeat( :, grpCls ) = probAll;
  end % end for each cluster

  % bayes combine
  simpleProb = BayesCombine( conf, imdb, curGrp, curModel );

  % compare
  fprintf( 'original: \n' );
  [ ~, trainAcc ] = ScoreToConf( curModel.bayesProb( train, : ), ...
    trainLab );
  fprintf( '\t train acc -- %.2f %%\n', trainAcc );
  [ ~, testAcc ] = ScoreToConf( curModel.bayesProb( test, : ), ...
    testLab );
  fprintf( '\t test acc -- %.2f %%\n', testAcc );

  fprintf( 'simple softmax: \n' );
  [ ~, trainAcc ] = ScoreToConf( simpleProb( train, : ), ...
    trainLab );
  fprintf( '\t train acc -- %.2f %%\n', trainAcc );
  [ ~, testAcc ] = ScoreToConf( simpleProb( test, : ), ...
    testLab );
  fprintf( '\t test acc -- %.2f %%\n', testAcc );
  
  if( g <= 5 )
    scores = scores + simpleProb;
  end
end % end for each group


[ ~, grpAcc ] = ScoreToConf( scores( test, : ), testLab );
fprintf( '\n Fuse Acc: %.2f %%\n', grpAcc );
