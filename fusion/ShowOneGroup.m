function [ clusterResult, clsResult, mapResult ] = ShowOneGroup( conf, curGrp, curModel )
%% ShowOneGroup
%  Desc: show one group model's performance
%  In: 
%    conf -- (stuct) configuration
%    curGrp -- (struct) group clustering information
%    curModel -- (struct) group model information
%  Out:
%    clusterAcc -- clusterring accuracy
%    clsAcc -- class accuracy
%%

% fprintf( 'function: %s\n', mfilename );


if( exist( 'curModel', 'var' ) && exist( 'curGrp', 'var' ) )
  load( conf.imdbPath );
  test = find( imdb.ttSplit == 0 );
  testClusterLab = curGrp.clusterLabel( test );
  testClsLab = imdb.clsLabel( test );
  % fprintf( 'current group and model:\n' );
  % disp( curGrp );
  % disp( curModel );
  
  [ clusterResult.conf, clusterResult.mA ] = ScoreToConf( curGrp.clusterScore, testClusterLab );
  fprintf( '\t Cluster accuracy: %.2f %%\n', clusterResult.mA );
  [ clsResult.conf, clsResult.mA ] = ScoreToConf( curModel.scores, testClsLab );
  fprintf( '\t Class accuracy: %.2f %%\n', clsResult.mA );
  [ mapResult.conf, mapResult.mA ] = ScoreToConf( curModel.mapFeat( test, : ), testClsLab );
  fprintf( '\t Map accuracy: %.2f %%\n', mapResult.mA  );
  
else
  fprintf( 'Error: missing curModel or curGrp var\' );
end

% end function ShowOneGroup