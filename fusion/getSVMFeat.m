%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: getSVMFeat.m
% Desc: get SVM features for one group
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
fprintf( '\t get mapping SVM features\n' );

curModel.mapSVM = cell( 1, numClasses );
curModel.mapFeat = zeros( numSample, numClasses );

for f = 1 : conf.foldNum
  fprintf( '\t\t Fold: %d (%.2f %%)\n', f, 100 * f / conf.foldNum );
  for c = 1 : grp{ g }.clusterNum
    fprintf( '\t\t Cluster: %d (%.2f %%)\n', c, 100 * c / grp{ g }.clusterNum );
    grpCls = grp{ g }.cluster{ c };
    for cmpCls = grpCls
      fprintf( '\t\t\t train test class: %d\n', cmpCls );
      tmpTrainIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
        cvTrain{ f } );
      tmpValidIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
        cvValid{ f } );
      if conf.isDebug
        % debug info
        fprintf( 'train len: %d valid len: %d\n', length( tmpTrainIdx ), length( tmpValidIdx ) );
      end
      grpTrainK = kernel( tmpTrainIdx, tmpTrainIdx );
      grpValidK = kernel( tmpValidIdx , tmpTrainIdx );

      grpTrainK = [ ( 1 : size( grpTrainK, 1 ) )', grpTrainK ];
      grpValidK = [ ( 1 : size( grpValidK, 1 ) )', grpValidK ];

      % train one-fold SVM
      yTrain = 2 * ( imdb.clsLabel( tmpTrainIdx ) == cmpCls ) - 1 ;
      yValid = 2 * ( imdb.clsLabel( tmpValidIdx ) == cmpCls ) - 1 ;
      tmpModel = libsvmtrain( double( yTrain ), double( grpTrainK ), ...
        conf.orgSVMOPT ) ;
      % get valid SVM score --> map features
      if( conf.isSVMProb )
        [gPrdCls, acc, tmpScore ] = libsvmpredict( double( yValid ), ...
          double( grpValidK ), tmpModel, '-b 1'  );
      else
        [gPrdCls, acc, tmpScore ] = libsvmpredict( double( yValid ), ...
          double( grpValidK ), tmpModel  );
      end
      if conf.isDebug 
        % debug info
        fprintf( 'tmpScore size: %d\n', size( tmpScore ) );
      end
      curModel.mapFeat( tmpValidIdx, cmpCls ) = tmpScore;

    end % end for grpCls
  end % end for cluster
end % end for fold

%-----------------------------------------------
% save current model to stage output
%-----------------------------------------------
save( cacheStage{ s }, 'curModel' );

fprintf( '\t get mapping SVM features time: %.2f (s)\n', toc );
