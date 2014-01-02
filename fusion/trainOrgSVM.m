%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: trainOrgSVM.m
% Desc: train original SVM to get test mapping features
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
fprintf( '\t train original SVM\n' );

curModel.orgSVM = cell( 1, numClasses );

for c = 1 : grp{ g }.clusterNum
  fprintf( '\t\tCluster: %d (%.2f %%)\n', c, 100 * c / grp{ g }.clusterNum );      
  grpCls = grp{ g }.cluster{ c };
  for cmpCls = grpCls
    fprintf( '\t\t train test class: %d\n', cmpCls );
    tmpTrainIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
        train );
    % use clustering results
    tmpTestIdx = intersect( find( grp{ g }.clsToCluster == c ), ...
        test );

    grpTrainK = kernel( tmpTrainIdx, tmpTrainIdx );
    grpTestK = kernel( tmpTestIdx, tmpTrainIdx );

    grpTrainK = [ ( 1 : size( grpTrainK, 1 ) )', grpTrainK ];
    grpTestK = [ ( 1 : size( grpTestK, 1 ) )', grpTestK ];

    % train original SVM
    yTrain = 2 * ( imdb.clsLabel( tmpTrainIdx ) == cmpCls ) - 1 ;
    yTest  = 2 * ( imdb.clsLabel( tmpTestIdx )  == cmpCls ) - 1 ;
    curModel.orgSVM{ cmpCls } = libsvmtrain( double( yTrain ), double( grpTrainK ), ...
      conf.orgSVMOPT ) ;
    % get test SVM score --> map features
    if( conf.isSVMProb )
      [gPrdCls, acc, tmpScore ] = libsvmpredict( double( yTest ), ...
        double( grpTestK ), curModel.orgSVM{ cmpCls }, '-b 1'  );
    else
      [gPrdCls, acc, tmpScore ] = libsvmpredict( double( yTest ), ...
        double( grpTestK ), curModel.orgSVM{ cmpCls }  );
    end
    curModel.mapFeat( tmpTestIdx, cmpCls ) = tmpScore;
  end % end for grpCls
end % end for cluster

%-----------------------------------------------
% save current model to stage output
%-----------------------------------------------
save( cacheStage{ s }, 'curModel' );

fprintf( '\t train original SVM time: %.2f (s)\n', toc );
