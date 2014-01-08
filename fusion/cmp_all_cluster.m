% compare all model vs cluster SVMs


test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
nClass = max( imdb.clsLabel );


for g = 1 : 8
  fprintf( 'group %d\n', g );
  cmpAcc = zeros( grpInfo{ g }.nCluster, 2 );
  for c = 1 : grpInfo{ g }.nCluster
    grpCls = grpInfo{ g }.cluster{ c };
    fprintf( '\t Cluster %d\n', c );
    % compare each cluster accuracy
    tmpIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
      test );
    otherIdx = setdiff( ( 1 : nClass ), grpCls );
    tmpGtLab = imdb.clsLabel( tmpIdx );
    
    allScores = grpModel{ 1 }.mapFeat( tmpIdx, : );
    curScores = grpModel{ g }.mapFeat( tmpIdx, : );
    
    allScores( :, otherIdx ) = -1e10;
    curScores( :, otherIdx ) = -1e10;
    
    [ ~, allLab ] = max( allScores, [], 2 );
    [ ~, curLab ] = max( curScores, [], 2 );
    
    cmpAcc( c, 1 ) = 100 * sum( allLab == tmpGtLab ) / length( tmpIdx );
    cmpAcc( c, 2 ) = 100 * sum( curLab == tmpGtLab ) / length( tmpIdx );
    fprintf( '\t org acc: %.2f %% -- g6 acc: %.2f %%\n', ...
      cmpAcc( c, 1 ), cmpAcc( c, 2 ) );
  end
  oldAcc( g ) = mean( cmpAcc( :, 1 ) );
  grpAcc( g ) = mean( cmpAcc( :, 2 ) );
  fprintf( 'mean org: %.2f %% -- mean group %d: %.2f %%\n', ...
    mean( cmpAcc( :, 1 ) ), g, mean( cmpAcc( :, 2 ) ) );
end
