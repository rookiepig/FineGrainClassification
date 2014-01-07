
lambdaRange = [ 0.01 0.1 0.3 1 3 9 10 ];

for l = 1 : length( lambdaRange )
  lambda = lambdaRange( l );
  fprintf( 'lambda: %f\n', lambda );
  regScores = TrainMapReg( conf, imdb, curModel.mapFeat, lambda  );
  [ regConf, regAcc ] = ScoreToConf( regScores, testLab );
  fprintf( 'Acc: %.2f\n', regAcc );
end
