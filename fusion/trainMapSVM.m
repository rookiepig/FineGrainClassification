function [ mapSVM, scores ] = TrainMapSVM( conf, imdb, mapFeat )
%% TrainMapSVM
%  Desc: train map SVM to get final score
%  In: 
%    conf, imdb -- basic variables
%    mapFeat -- (nSample * nClass) SVM feature with test SVM feat
%  Out:
%    mapSVM  -- map SVM model
%    scores  -- (nSample * nClass) final score for map SVM
%%

fprintf( 'function: %s\n', mfilename );
tic;

% init basic variables
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
train   = find( imdb.ttSplit == 1 );
test    = find( imdb.ttSplit == 0 );

mapSVM = cell( 1, nClass );
scores = cell( 1, nClass );

% without SVM prob --> normalize all map features
if( ~conf.isSVMProb )
  fprintf( 'normalize map features\n' );
  for m = 1 : nSample
    z = mapFeat( m, : );
    % find value equal to MAP_INIT_VAl
    initIdx  = find( z == conf.MAP_INIT_VAL );
    otherIdx = find( z ~= conf.MAP_INIT_VAL );
    z( initIdx ) = -0.1;
    z( otherIdx ) = ( z( otherIdx ) - min( z( otherIdx) ) ) ./ ...
      max( max( z( otherIdx ) ) - min( z( otherIdx ) ), 1e-12 );
    mapFeat( m, : ) = z;
  end
end
for c = 1 : nClass
  % train map SVM
  fprintf( '\t train test map SVM class: %d (%.2f %%)\n', c, 100 * c / nClass );
  yTrain = 2 * ( imdb.clsLabel( train ) == c ) - 1 ;
  yTest  = 2 * ( imdb.clsLabel( test )  == c ) - 1 ;
  mapSVM{ c } = libsvmtrain( double( yTrain ), double( mapFeat( train, : ) ), ...
    conf.mapSVMOPT ) ;
  % get final score
  [gPrdCls, acc, scores{ c } ] = libsvmpredict( double( yTest ), ...
    double( mapFeat( test, : ) ), mapSVM{ c }  );
end % end for class

scores = cat( 2, scores{ : } );

fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainMapSVM
