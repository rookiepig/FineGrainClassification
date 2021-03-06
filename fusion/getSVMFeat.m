function [ mapFeat ] = GetSVMFeat( conf, imdb, kernel, curGrp )
%% GetSVMFeat
%  Desc: get training SVM feat use n-fold cross validation
%  In: 
%    conf, imdb, kernel -- basic variables
%    curGrp -- (struct) group clustering information
%  Out:
%    mapFeat -- (1 * nCluster) cell with (nSample * nClass) mapped SVM feature
%%

PrintTab();fprintf( 'function: %s\n', mfilename );
tic;

% init basic variables
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
nCluster = curGrp.nCluster;

% split 10 fold CV
[ cvTrain, cvValid ] = SplitCVFold( conf.nFold, imdb.clsLabel, ...
  imdb.ttSplit );

mapFeat = cell( 1, nCluster );
for c = 1 : nCluster
  mapFeat{ c } = zeros( nSample, nClass ) + conf.MAP_INIT_VAL;
end

for f = 1 : conf.nFold
  PrintTab();fprintf( '  Fold: %d (%.2f %%)\n', f, 100 * f / conf.nFold );
  for c = 1 : nCluster
    PrintTab();
    fprintf( '    Cluster: %d (%.2f %%)\n', c, 100 * c / nCluster );
    grpCls = curGrp.cluster{ c };
    for gC = 1 : length( grpCls )
      cmpCls = grpCls( gC );
      fprintf( '*.' );
      % init train valid index
      trainIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
        cvTrain{ f } );
      if( conf.useClusterPrior )
        % use cluster information 
        validIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
          cvValid{ f } );
      else
        % all validation are tested on cluster SVM
        validIdx = cvValid{ f };
      end

      trainK = kernel( trainIdx, trainIdx );
      validK = kernel( validIdx , trainIdx );

      trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
      validK = [ ( 1 : size( validK, 1 ) )', validK ];

      % train one-fold SVM
      yTrain = 2 * ( imdb.clsLabel( trainIdx ) == cmpCls ) - 1 ;
      yValid = 2 * ( imdb.clsLabel( validIdx ) == cmpCls ) - 1 ;
      tmpModel = libsvmtrain( double( yTrain ), double( trainK ), ...
        curGrp.grpSVMOPT );
        %conf.orgSVMOPT ) ;

      % get valid SVM score --> map features
      [ ~, ~, tmpScore ] = libsvmpredict( double( yValid ), ...
        double( validK ), tmpModel  );

      mapFeat{ c }( validIdx, cmpCls ) = tmpScore;
    end % end for grpCls
    fprintf( '\n' );
  end % end for cluster
end % end for fold

PrintTab();fprintf( '\t function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function GetSVMFeat
