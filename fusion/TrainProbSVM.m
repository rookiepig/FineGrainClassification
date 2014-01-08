function [ probFeat, probSVM ] =  TrainProbSVM( conf, imdb, kernel, curGrp )
%% TrainProbSVM
%  Desc: train each class SVM for each cluster using libsvm probability
%  In: 
%    conf, imdb, kernel -- basic variables
%    curGrp -- (struct) group clustering information
%  Out:
%    probFeat -- (nSample * nClass) probability output for all samples
%    probSVM  -- (nCluster * 1) class SVM model for each cluster
%%

fprintf( '\t function: %s\n', mfilename );
tic;

% init basic variables
nClass  = max( imdb.clsLabel );
nCluster = curGrp.nCluster;
nSample = length( imdb.clsLabel );

train   = find( imdb.ttSplit == 1 );
test    = find( imdb.ttSplit == 0 );

% one proSVM for each cluster
probSVM  = cell( 1, nCluster );
probFeat = zeros( nSample, nClass );

for c = 1 : nCluster
  fprintf( '\t Cluster: %d (%.2f %%)\n', c, 100 * c / nCluster );      
  grpCls = curGrp.cluster{ c };
  if( length( grpCls ) == 1  )
    % only one class in cluster
    fprintf( '\t Warning: only one class no need to train\n' );
    probSVM{ c } = [];
    probFeat( :, grpCls( 1 ) ) = 1;
  else
    clutsterTrain = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
        train );
    % prepare kernel
    trainK = kernel( clutsterTrain, clutsterTrain );
    allK = kernel(  : , clutsterTrain );
    trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
    allK = [ ( 1 : size( allK, 1 ) )', allK ];

    % train multi-class probability model for each cluster
    yTrain = imdb.clsLabel( clutsterTrain );
    yAll  = imdb.clsLabel;
    % train
    probSVM{ c } = libsvmtrain( double( yTrain ), double( trainK ), ...
      conf.orgSVMOPT );
    % test
    [ ~, ~, tmpProb ] = libsvmpredict( double( yAll ), ...
      double( allK ), probSVM{ c }, '-b 1'  );
    % use Label to map tmpProb --> probFeat
    for t = 1 : probSVM{ c }.nr_class
      probFeat( :, probSVM{ c }.Label( t ) ) = tmpProb( :, t );
    end
  end % end if( length( grpCls ) == 1  )
end % end for cluster

fprintf( '\t function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainProbSVM
