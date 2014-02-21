%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: overlap_cluster_acc.m
% Desc: overlap cluster accuracy
% Author: Zhang Kang
% Date: 2014/02/16
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load( 'data/CUB11/imdb.mat' );

if( ~exist( 'curGrp', 'var' ) )
  fprintf( 'Error: var curGrp not exist\n');
else
  test = find( imdb.ttSplit == 0 );
  testLab = imdb.clsLabel( test );
  correct = 0;
  for t = 1 : length( test )
    c = curGrp.clsToCluster( test( t ) );
    if( ismember( testLab( t ), curGrp.cluster{ c } ) )
      correct = correct + 1;
    end
  end
  fprintf( 'test (overlap) cluster acc: %.2f %%\n', 100 * correct / length( test ) );
end
