meanAcc = 0;
for c = 1 : curGrp.nCluster
  fprintf( '\t Cluster: %d (%.2f %%)\n', c, 100 * c / curGrp.nCluster );   
  % use clustering results
  testIdx = intersect( find( curGrp.clsToCluster == c ), ...
    test );   
  grpCls = curGrp.cluster{ c };
  tmpScore = zeros( length( testIdx ), length( grpCls ) );
  tmpLabel = imdb.clsLabel( testIdx );
  tmpGtLabel = floor( rand( size( tmpLabel ) ) * length( grpCls ) + 1 );
  for gC = 1 : length( grpCls )
    cmpCls = grpCls( gC );
    tmpIdx = find( tmpLabel == cmpCls );
    tmpGtLabel( tmpIdx ) = gC;
    fprintf( '\t\t train test class: %d\n', cmpCls );
    tmpScore( :, gC ) = curModel.mapFeat( testIdx, cmpCls );
  end % end for grpCls
  [ tmpConf, tmpAcc ] = ScoreToConf( tmpScore, tmpGtLabel );
  fprintf( '\t cluster mA: %.2f %%\n', tmpAcc );
  meanAcc = meanAcc + tmpAcc;
  fprintf( 'Press any key to continue..\n' );
  pause;
end % end for cluster
meanAcc = meanAcc / curGrp.nCluster;
fprintf( 'All cluster mean mA: %.2f %%\n', meanAcc );



