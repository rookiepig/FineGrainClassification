numClasses = numel( imdb.clsName );
train = ( imdb.ttSplit == 1 );
test = ( imdb.ttSplit == 0 ) ;

trainScores = cell( 1, numClasses );

% get training scores
for c = 1 : numClasses
  fprintf( '\n\t get training scores: %s (%.2f %%)\n', ...
    imdb.clsName{ c }, 100 * c / numClasses );
  % one-vs-rest SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  [predClass, acc, trainScores{ c } ] = libsvmpredict( double( y( train ) ), ...
    double( kernelTrain ), model{ c } );
end
trainScores = cat( 2, trainScores{ : } );


% show some examples
subplot( 2, 2, 1 ); bar( trainScores( 1, : ), 'BaseValue', - 1 );
subplot( 2, 2, 2 ); bar( trainScores( 2, : ), 'BaseValue', - 1 );
subplot( 2, 2, 3 ); bar( trainScores( 3, : ), 'BaseValue', - 1 );
subplot( 2, 2, 4 ); bar( trainScores( 4, : ), 'BaseValue', - 1 );


% get most confused class for each class
for c = 1 : numClasses
  curConf = confusion( c, : );
  fprintf( ' Class: %s\n', imdb.clsName{ c } );
  fprintf( '\t precision: %.2f %%\n', 100 * curConf( c ) );
  curConf( c ) = 0;
  [ errPrecs( c ), errCls( c ) ] = max( curConf );
  fprintf( '\t error precision: %.2f %%\n', 100 * errPrecs( c ) );
  fprintf( '\t error class: %s\n', imdb.clsName{ errCls( c ) } );
  % pause;
end

% get 10 worst and 10 best classes
for c = 1 : 10
  fprintf( '%-25s , (%.2f %%) , %-25s , (%.2f %%) \n', imdb.clsName{ b( c ) }, 100 * a( c ), ...
    imdb.clsName{ errCls( b( c ) ) }, 100 * errPrecs( b( c ) ) );
end


% compare group SVM vs all SVM
cmpCls = 71;
groupCls = [ 71 72 ];

groupIdx = ismember( imdb.clsLabel, groupCls );
groupTrain = train & groupIdx;
groupTest = test & groupIdx;
groupTrainKernel = kernelTrain( groupTrain, groupTrain );
groupTestKernel = kernelTest( : , groupTrain );
for c = cmpCls : cmpCls
  fprintf( '\n\t group train class: %s (%.2f %%)\n', ...
    imdb.clsName{ c }, 100 * c / numClasses );
  % group SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  groupModel = libsvmtrain( double( y( groupTrain ) ), double( groupTrainKernel ), ...
    '-c 10 -t 4' ) ;
  [groupPredCls, acc, groupScore ] = libsvmpredict( double( y( test ) ), ...
    double( groupTestKernel ), groupModel );
  [~,~,info] = vl_pr( y( test ), groupScore ) ;
  groupAP = info.ap ;
  groupAP11 = info.ap_interp_11 ;
  fprintf('\n\t group: AP %.2f; AP 11 %.2f\n', ...
    groupAP * 100, groupAP11 * 100 ) ;
  % all SVM
  [allPredCls, acc, allScore ] = libsvmpredict( double( y( test ) ), ...
    double( kernelTest ), model{ c } );
  [~,~,info] = vl_pr( y( test ), allScore ) ;
  allAP = info.ap ;
  allAP11 = info.ap_interp_11 ;
  fprintf('\n\t all: AP %.2f; AP 11 %.2f\n',  ...
    allAP * 100, allAP11 * 100 ) ;
end















