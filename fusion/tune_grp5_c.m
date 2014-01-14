% tune c for group 5

selC = [ 1000 3000 10000 30000 ];

for t = 1 : length( selC )
  curC = selC( t );
  fprintf( 'current C = %f\n', curC );
  % set grp svm opt
  curGrp.grpSVMOPT = sprintf( '-c %f -t 4 -q', curC );
  % get curModel
  curModel = TrainGroupModel( curGrp );
  % compare one group

  cmpAcc = zeros( curGrp.nCluster, 2 );
  for c = 1 : curGrp.nCluster
    grpCls = curGrp.cluster{ c };
    PrintTab();fprintf( '\t Cluster %d\n', c );
    % compare each cluster accuracy
    tmpIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
      test );
    otherIdx = setdiff( ( 1 : nClass ), grpCls );
    tmpGtLab = imdb.clsLabel( tmpIdx );

    allScores = orgModel.svmScore( tmpIdx, : );
    curScores = curModel.svmScore( tmpIdx, : );

    allScores( :, otherIdx ) = -1e10;
    curScores( :, otherIdx ) = -1e10;

    [ ~, allLab ] = max( allScores, [], 2 );
    [ ~, curLab ] = max( curScores, [], 2 );

    cmpAcc( c, 1 ) = 100 * sum( allLab == tmpGtLab ) / length( tmpIdx );
    cmpAcc( c, 2 ) = 100 * sum( curLab == tmpGtLab ) / length( tmpIdx );
    PrintTab();fprintf( '\t org acc: %.2f %% -- grp acc: %.2f %%\n', ...
      cmpAcc( c, 1 ), cmpAcc( c, 2 ) );
  end
  oldAcc( t ) = mean( cmpAcc( :, 1 ) );
  grpAcc( t ) = mean( cmpAcc( :, 2 ) );
  PrintTab();fprintf( 'mean org: %.2f %% -- mean group 5: %.2f %%\n', ...
    mean( cmpAcc( :, 1 ) ),  mean( cmpAcc( :, 2 ) ) );

end