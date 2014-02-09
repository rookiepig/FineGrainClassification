function [ curGrp ] = TrainClusterModel( curGrp )
%% TrainClusterModel
%  Desc: train cluster model --> get test cluster label
%  In: 
%    curGrp -- (struct) clustering infomation for one group
%  Out:
%    curGrp -- (struct) 
%      - curGrp.nCluster  - number of clusters
%      - curGrp.cluster      - (1 * nCluster) class index
%      - clusterGtLab - (nSample * 1) ground truth cluster label 
%      - curGrp.clusterSVM   - (1 * nCluster) SVM model
%      - curGrp.testScore - (nSample * cluterNum) SVM score
%      - curGrp.testConf  - (nCluster * nCluster) confusion matrix
%      - curGrp.clsToCluster - (nSample * 1) cluster for each sample
%%

PrintTab();fprintf( 'function: %s\n', mfilename );
tID = tic;

% init basic variables
conf = InitConf( );
load( conf.imdbPath );
PrintTab();fprintf( '\t loading kernel (maybe slow)\n' );
load( conf.kernelPath );
nSample = length( imdb.clsLabel );
train = find( imdb.ttSplit == 1 );
test  = find( imdb.ttSplit == 0 );
clsNum = curGrp.nCluster;

% get cluster label
clusterGtLab = zeros( size( imdb.clsLabel ) );
for k = 1 : curGrp.nCluster
  clusterIdx = find( ismember( imdb.clsLabel, curGrp.cluster{ k } ) );
  clusterGtLab( clusterIdx ) = k;
end
% save to curGrp
curGrp.clusterGtLab = clusterGtLab;

% record libsvm probability option!
curGrp.isOVOSVM = conf.isOVOSVM;

if( clsNum == 1 )

  % one cluster no need to train
  PrintTab();
  fprintf( '\t Warning: only one cluster no need to train\n' );
  curGrp.clusterSVM = [];
  curGrp.clsToCluster = clusterGtLab;
  curGrp.clusterProb  =  ones( size( imdb.clsLabel ) );

else

  if( conf.isOVOSVM )
    %% use one-vs-one libsvm prob
    PrintTab();fprintf( '\t use libsvm one-vs-one probability\n' );
    % init training and testing kernel
    trainK = kernel( train, train );
    trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
    allK  = kernel( : , train );
    allK = [ ( 1 : size( allK, 1 ) )', allK ];
    yTrain = clusterGtLab( train );
    yAll  = clusterGtLab;
    % train libsvm (with probability)
    curGrp.clusterSVM = libsvmtrain( double( yTrain ), ...
      trainK, conf.clusterSVMOPT );
    % test
    [ tmpPred, ~, tmpProb ] = libsvmpredict( double( yAll ), ...
      allK, curGrp.clusterSVM, '-b 1' );
    % get all predicted clusters
    curGrp.clsToCluster = tmpPred;
    % need to remap probability due to class inverse older
    for t = 1 : curGrp.clusterSVM.nr_class
      curGrp.clusterProb( :, curGrp.clusterSVM.Label( t ) ) = ...
        tmpProb( :, t );
    end

  else

    %% n-fold CV get train cluster scores
    PrintTab();
    fprintf( '\t %d-fold CV one-vs-all cluster SVM\n', conf.nFold );
    % get train valid index
    [ cvTrain, cvValid ] = SplitCVFold( conf.nFold, ...
      clusterGtLab, imdb.ttSplit );
    % using n-fold CV to get training sample cluster score
    curGrp.clusterScore = zeros( nSample, clsNum );
    for f = 1 : conf.nFold
      PrintTab();
      fprintf( '\t Fold: %d (%.2f %%)\n', f, 100 * f / conf.nFold );
      % init train valid index
      trainIdx = cvTrain{ f };
      validIdx = cvValid{ f };
      % prepare train and valid kernel
      trainK = kernel( trainIdx, trainIdx );
      validK = kernel( validIdx , trainIdx );
      trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
      validK = [ ( 1 : size( validK, 1 ) )', validK ];
      % train classifier
      for c = 1 : clsNum
        fprintf( '.*' );
        yTrain = 2 * ( clusterGtLab( trainIdx ) == c ) - 1;
        yValid = 2 * ( clusterGtLab( validIdx ) == c ) - 1;
        % train
        tmpSVM = libsvmtrain( double( yTrain ), ...
          double( trainK ), conf.clusterSVMOPT );
        % validation score
        [ ~,~, tmpScore ] = libsvmpredict( double( yValid ), ...
          double( validK ), tmpSVM );
        curGrp.clusterScore( validIdx, c ) = tmpScore;
      end
      fprintf( '\n' );
    end % end each fold

    %% all train sample to get test cluster scores
    % init training and testing kernel
    trainK = kernel( train, train );
    trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
    testK  = kernel( test, train );
    testK = [ ( 1 : size( testK, 1 ) )', testK ];
    % train and test cluster SVM
    curGrp.clusterSVM = cell( 1, clsNum );
    % get test cluster label
    for c = 1 : clsNum
      PrintTab();
      fprintf( '\t cluster train test: %d (%.2f %%)\n', c, 100 * c / clsNum );
      y = 2 * ( clusterGtLab == c ) - 1;
      % train
      curGrp.clusterSVM{ c } = libsvmtrain( double( y( train ) ), ...
        double( trainK ), conf.clusterSVMOPT );
      % test
      [ ~,~, tmpScore ] = libsvmpredict( double( y( test ) ), ...
        double( testK ), curGrp.clusterSVM{ c } );
      curGrp.clusterScore( test, c ) = tmpScore;
    end

    %% cluster score mapping
    PrintTab();fprintf( '\t cluster score map type %s\n', conf.mapType );
    curGrp.mapType = conf.mapType;
    switch conf.mapType
      case 'reg'
        % kernel regression
        curGrp.clusterScore = NormMapFeat( conf, imdb, curGrp.clusterScore );
        curGrp.clusterProb  = TrainMapReg( conf, imdb, ...
          curGrp.clusterScore, clusterGtLab );
      case 'softmax'
        % softmax regression --> probability
        [ wSoftmax, proAll ] = MultiLRL2( curGrp.clusterScore( train, : ), ...
                                          clusterGtLab( train ), ...
                                          curGrp.clusterScore, 1, ones( length( train ), 1 ) );
        curGrp.wSoftmax = wSoftmax;
        curGrp.clusterProb = proAll;
      otherwise
        PrintTab();fprintf( 'Error: unknown map type: %s\n', conf.mapType );
    end

    % set predicted cluster label
    [ ~, trainPred ] = max( curGrp.clusterProb( train, : ), [], 2 );
    [ ~, testPred ] = max( curGrp.clusterProb( test, : ), [], 2 );
    curGrp.clsToCluster = zeros( size( imdb.clsLabel ) );
    curGrp.clsToCluster( train ) = trainPred;
    curGrp.clsToCluster( test ) = testPred;
  end % end if( isOVOSVM )

end % end if( clsNum == 1 )

% get confusion matrix
[ curGrp.trainConf, curGrp.trainAcc ] = ...
  ScoreToConf( curGrp.clusterProb( train, : ), clusterGtLab( train ) );
PrintTab();
fprintf( '\t train cluster mA: %.2f %%\n', curGrp.trainAcc );
[ curGrp.testConf, curGrp.testAcc ] = ...
  ScoreToConf( curGrp.clusterProb( test, : ), clusterGtLab( test ) );
PrintTab();
fprintf( '\t test cluster mA: %.2f %%\n', curGrp.testAcc );

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc( tID ) );

% end function TrainClusterModel