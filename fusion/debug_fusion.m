


numClasses = numel( imdb.clsName );
train = find( imdb.ttSplit == 1 );
test = find( imdb.ttSplit == 0 ) ;
GRP_SYS_NUM = numel( grp );

%% split 10 fold training and validation set
FOLD_NUM = 10;
cvTrain = cell( 1, FOLD_NUM );
cvValid = cell( 1, FOLD_NUM );

%% mapping SVM and original SVM
for g = 8 : 8
  %
  % train map SVM & get final score
  % 
  tic;
  fprintf( '\t train map SVM and get final scores\n' );
  grp{ g }.scores = cell( 1, numClasses );
  for c = 1 : numClasses
    % train map SVM
    yTrain = 2 * ( imdb.clsLabel( train ) == c ) - 1 ;
    yTest  = 2 * ( imdb.clsLabel( test )  == c ) - 1 ;
    grp{ g }.mapSVM{ c } = libsvmtrain( double( yTrain ), double( grp{ g }.mapFeat( train, : ) ), ...
      '-s 2 -c 10 -t 2' ) ;
    % get final score
    [gPrdCls, acc, grp{ g }.scores{ c } ] = libsvmpredict( double( yTest ), ...
      double( grp{ g }.mapFeat( test, : ) ), grp{ g }.mapSVM{ c }  );
  end % end for class
  fprintf( '\t train map SVM and get final scores time: %.2f (s)\n', toc );
  if DEBUG
    % save temp model
    fprintf( 'Save temp model: %s\n', tmpModelName );
    save( tmpModelName, 'grp' );
  end 
end % end for group