%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: trainMapSVM.m
% Desc: train mapping SVM to get final score
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
fprintf( '\t train map SVM and get final scores\n' );

curModel.scores = cell( 1, numClasses );
% without SVM prob --> normalize all map features
if( ~conf.isSVMProb )
  fprintf( '\t normalize map features\n' );
  for m = 1 : numSample
    z = curModel.mapFeat( m, : );
    z = ( z - min( z ) ) ./ max( max( z ) - min( z ), 1e-12);
    curModel.mapFeat( m, : ) = z;
  end
end
for c = 1 : numClasses
  % train map SVM
  fprintf( '\t train map SVM class: %d (%.2f %%)\n', c, 100 * c / numClasses );
  yTrain = 2 * ( imdb.clsLabel( train ) == c ) - 1 ;
  yTest  = 2 * ( imdb.clsLabel( test )  == c ) - 1 ;
  curModel.mapSVM{ c } = libsvmtrain( double( yTrain ), double( curModel.mapFeat( train, : ) ), ...
    conf.mapSVMOPT ) ;
  % get final score
  [gPrdCls, acc, curModel.scores{ c } ] = libsvmpredict( double( yTest ), ...
    double( curModel.mapFeat( test, : ) ), curModel.mapSVM{ c }  );
end % end for class

%-----------------------------------------------
% save current model to stage output
%-----------------------------------------------
save( cacheStage{ s }, 'curModel' );

fprintf( '\t train map SVM and get final scores time: %.2f (s)\n', toc );
