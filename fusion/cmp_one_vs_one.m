% get original one-vs-one results
tic;

% init configuration
conf = InitConf( );

% load imdb, kernel
load( conf.imdbPath );
if( ~exist( 'kernel', 'var' ) )
  fprintf( 'loading kernel (maybe slow)\n' );
  load( conf.kernelPath );
end


% init basic variables
nClass  = max( imdb.clsLabel );
nSample = length( imdb.clsLabel );

train   = find( imdb.ttSplit == 1 );
test    = find( imdb.ttSplit == 0 );


% prepare kernel
trainK = kernel( train, train );
testK  = kernel( test, train );

trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
testK = [ ( 1 : size( testK, 1 ) )', testK ];

% one-vs-one classifier
fprintf( 'one-vs-one SVM\n' );
yTrain = imdb.clsLabel( train );
yTest  = imdb.clsLabel( test );

cParam = [ 10 ];

for p = 1 : length( cParam )
  svmOpt = sprintf( '-c %f -t 4 -q', cParam( p ) );
  % train model
  fprintf( 'svm param: %s\n', svmOpt );
  tmpModel = libsvmtrain( double( yTrain ), double( trainK ), svmOpt );
  % train acc
  [ tmpPred, tmpAcc, tmpDec ] = libsvmpredict( double( yTrain ), ...
    double( trainK ), tmpModel );
  % confusion
  confusion = confusionmat( yTrain, tmpPred );
  for c = 1 : nClass
    sumC = sum( confusion( c , : ) );
    confusion( c, : ) = confusion( c, : ) ./ max( sumC, 1e-12 );
  end
  ovoTrainAcc = 100 * mean(diag(confusion));
  fprintf( '\n C = %f -- train mean acc: %.2f %%\n', cParam( p ), ovoTrainAcc );

  % test
  [ tmpPred, tmpAcc, tmpDec ] = libsvmpredict( double( yTest ), ...
    double( testK ), tmpModel );
  % confusion
  confusion = confusionmat( yTest, tmpPred );
  for c = 1 : nClass
    sumC = sum( confusion( c , : ) );
    confusion( c, : ) = confusion( c, : ) ./ max( sumC, 1e-12 );
  end
  ovoTestAcc = 100 * mean(diag(confusion));
  fprintf( '\n C = %f -- test mean acc: %.2f %%\n', cParam( p ), ovoTestAcc );
end % tune C

fprintf( 'one-vs-one SVM train test time: %.2f (s)\n', toc );

% one-vs-all classifier
fprintf( 'one-vs-all SVM\n' );
for c = 1 : nClass
  fprintf( '\n\t training class: %s (%.2f %%)\n', ...
    imdb.clsName{ c }, 100 * c / nClass );
  % one-vs-rest SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  model{ c } = libsvmtrain( double( y( train ) ), double( trainK ), ...
    '-c 10 -t 4 -q' ) ;
  [predClass, acc, scores{ c } ] = libsvmpredict( double( y( test ) ), ...
    double( testK ), model{ c } );
end

scores = cat(2,scores{:}) ;
% confusion matrix
[~,preds] = max(scores, [], 2) ;
confusion = confusionmat( imdb.clsLabel( test ), preds );
for c = 1 : nClass
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) / sumC;
end
ovaTestAcc = 100 * mean(diag(confusion));
fprintf( '\n one-vs-all mean acc: %.2f %%\n', ovaTestAcc );

