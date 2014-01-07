% improve cluster label model
% use cluster prior to 

% conf = InitConf();
load( conf.imdbPath );
load( conf.grpInfoPath );
load( conf.grpModelPath );

test = find( imdb.ttSplit == 0 );

% combine all groups' feature
% allFeat = zeros( size( grpModel{ 1 }.mapFeat ) );
for g = 2 : 6
  fprintf( 'Group %d\n', g );
  fprintf( 'old acc: %.2f %%\n', grpInfo{ g }.testAcc );
  nClass = grpInfo{ g }.nCluster;
  sampleLab = grpInfo{ g }.clusterGtLab;
  testLab = sampleLab( test );
  tryFeat =  grpInfo{ g }.clusterScore;
  tryFeat   = NormMapFeat( conf, imdb, tryFeat );
  tryScores = TrainMapReg( conf, imdb, tryFeat, sampleLab );
  
  [ tryConf{ g }, tryMeanAcc( g ) ] = ScoreToConf( tryScores, testLab );
  fprintf( '\t group %d -- Mean Acc: %.2f %%\n', g, tryMeanAcc( g ) );
end

