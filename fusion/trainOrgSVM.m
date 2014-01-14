function [ mapFeat, orgSVM ] =  TrainOrgSVM( conf, imdb, kernel, ...
  curGrp, mapFeat )
%% TrainOrgSVM
%  Desc: train one-vs-all SVM to get test svm scores
%  In: 
%    conf, imdb, kernel -- basic variables
%    curGrp -- (struct) group clustering information
%  Out:
%    mapFeat -- (nSample * nClass) SVM feature with test SVM feat
%    orgSVM  -- original SVM model
%%

PrintTab();fprintf( 'function: %s\n', mfilename );
tic;

%% different svm.C for different cluster
curSVMOPT = sprintf( '-c %f -t 4 -q', 10 * curGrp.nCluster );


% init basic variables
nClass  = max( imdb.clsLabel );
train   = find( imdb.ttSplit == 1 );
test    = find( imdb.ttSplit == 0 );

orgSVM = cell( 1, nClass );

for c = 1 : curGrp.nCluster
  PrintTab();
  fprintf( '\t Cluster: %d (%.2f %%)\n', c, 100 * c / curGrp.nCluster );      
  grpCls = curGrp.cluster{ c };
  for gC = 1 : length( grpCls )
    cmpCls = grpCls( gC );
    PrintTab();
    fprintf( '\t\t train test class: %d\n', cmpCls );
    trainIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
        train );
    if( conf.useClusterPrior )
      % use clustering prior
      testIdx = intersect( find( curGrp.clsToCluster == c ), ...
        test );
    else
      testIdx = test;
    end
    
    % prepare kernel
    trainK = kernel( trainIdx, trainIdx );
    testK = kernel( testIdx, trainIdx );

    trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
    testK = [ ( 1 : size( testK, 1 ) )', testK ];

    % train original SVM
    yTrain = 2 * ( imdb.clsLabel( trainIdx ) == cmpCls ) - 1 ;
    yTest  = 2 * ( imdb.clsLabel( testIdx )  == cmpCls ) - 1 ;
    orgSVM{ cmpCls } = libsvmtrain( double( yTrain ), double( trainK ), ...
      curGrp.grpSVMOPT );
      % conf.orgSVMOPT ) ;
    % get test SVM score --> map features
    [ ~, ~, tmpScore ] = libsvmpredict( double( yTest ), ...
      double( testK ), orgSVM{ cmpCls }  );

    mapFeat( testIdx, cmpCls ) = tmpScore;
  end % end for grpCls
end % end for cluster

PrintTab();fprintf( '\t function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainOrgSVM
