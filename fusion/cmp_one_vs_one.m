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
% yTrain = imdb.clsLabel( train );
% yTest  = imdb.clsLabel( test );

% cParam = [ 300, 1000, 3000, 10000, 30000 ];

% for p = 1 : length( cParam )
%   svmOpt = sprintf( '-c %f -t 4 -q', cParam( p ) );
%   % train
%   fprintf( 'svm param: %s\n', svmOpt );
%   tmpModel = libsvmtrain( double( yTrain ), double( trainK ), svmOpt );
%   % test
%   [ tmpPred, tmpAcc, tmpDec ] = libsvmpredict( double( yTest ), ...
%     double( testK ), tmpModel );
%   % confusion
%   confusion = confusionmat( yTest, tmpPred );
%   for c = 1 : nClass
%     sumC = sum( confusion( c , : ) );
%     confusion( c, : ) = confusion( c, : ) ./ max( sumC, 1e-12 );
%   end
%   meanAcc = 100 * mean(diag(confusion));
%   fprintf( '\n C = %f -- mean acc: %.2f %%\n', cParam( p ), meanAcc );
% end % tune C


% one-vs-all classifier
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
meanAcc = 100 * mean(diag(confusion));
fprintf( '\n one-vs-all mean acc: %.2f %%\n', meanAcc );

fprintf( 'one-vs-one SVM train test time: %.2f (s)\n', toc );

% end function TrainProbSVM
