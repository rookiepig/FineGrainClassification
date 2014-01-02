%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: genConfusion.m
% Desc: get confusion matrix for Group Fusion
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert each group's scores to matrix form
tmpGrpScores = cell( 1, GRP_SYS_NUM );
for g = 1 : GRP_SYS_NUM
  tmpGrpScores{ g } = cat( 2, grpModel{ g }.scores{ : } );
end

if( conf.useClusterPrior )
  % set non-cluster scores to negative infinite
  testNum = length( test );
  for g = 1 : GRP_SYS_NUM
    cTc = grp{ g }.clsToCluster( test );
    % set other cluster score to minimum
    for t = 1 : testNum
      clsIdx = grp{ g }.cluster{ cTc( t ) };
      clsIdx = setdiff( ( 1 : numClasses )', clsIdx );
      tmpGrpScores{ g }( t, clsIdx ) = -1e10;
    end
  end
end

% average SVM scores
fprintf( 'Average SVM scores\n' );
avg_scores = zeros( size( tmpGrpScores{ 1 } ) );
for g = 1 : GRP_SYS_NUM
  avg_scores = avg_scores + tmpGrpScores{ g };
end
avg_scores = avg_scores ./ GRP_SYS_NUM;

% get confusion matrix
fprintf( 'Get confusion matrix\n' );
% confusion matrix
[~,preds] = max(avg_scores, [], 2) ;
confusion = confusionmat( imdb.clsLabel( test ), preds );
for c = 1 : numClasses
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) / sumC;
end
meanAccuracy = sprintf('mean accuracy: %.2f %%\n', 100 * mean(diag(confusion)));
figure(1) ; clf ;
imagesc(confusion) ; axis square ;
title(meanAccuracy) ;
vl_printsize(1) ;
print('-dpdf', fullfile( conf.outDir, [ conf.prefix, 'confusion.pdf' ] ) );
%print('-djpeg', fullfile(conf.outDir, 'result-confusion.jpg')) ;

% save results
save( conf.resultPath, 'conf', 'avg_scores', 'meanAccuracy', 'confusion' );

