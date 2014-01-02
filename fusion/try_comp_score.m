%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: try_comp_score.m
% Desc: compute mA using ground truth cluster
% Author: Zhang Kang
% Date: 2013/12/31
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load( 'grp_model.mat' );
load( 'imdb.mat' );

GRP_SYS_NUM  = numel( grp );
test = ( imdb.ttSplit == 0 );
testNum = sum( test );
numClasses = numel( imdb.clsName );

% calculate each group's mA
for g = 1 : GRP_SYS_NUM
  fprintf( 'Group: %d\n', g );
  scores = grp{ g }.scores;
  scores = cat( 2, scores{ : } );
  cTc = grp{ g }.clsToCluster( test );
  % set other cluster score to minimum
  for t = 1 : testNum
    clsIdx = grp{ g }.cluster{ cTc( t ) };
    clsIdx = setdiff( ( 1 : numClasses )', clsIdx );
    scores( t, clsIdx ) = -1e10;
  end
  % get confusion and mean accuracy
  [~,preds] = max(scores, [], 2) ;
  confusion = confusionmat( imdb.clsLabel( test ), preds );
  for c = 1 : numClasses
    sumC = sum( confusion( c , : ) );
    confusion( c, : ) = confusion( c, : ) / sumC;
  end
  meanAccuracy = sprintf('mean accuracy: %.2f %%\n', 100 * mean(diag(confusion)));
  figure(1) ; clf ;
  imagesc(confusion) ; axis square ;
  title([meanAccuracy]) ;
  pause;
  fprintf( 'Press any key to continue...\n' );
end

