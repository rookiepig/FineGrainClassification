%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: kernel_softmax_lite.m
% Desc: kernel multinomial l2 regularizatioin (lite version only 10 class)
% Author: Zhang Kang
% Date: 2014/01/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% init basic vars
conf = InitConf();
load( conf.imdbPath );
if( ~exist( 'kernel', 'var' ) )
  fprintf( 'load kernel %s\n' );
  load( conf.kernelPath );
end

% display iteration info
options.Display = 1;

liteIdx = find( imdb.clsLabel <= 10 );

test = intersect( liteIdx, find( imdb.ttSplit == 0 ) );
testLab = imdb.clsLabel( test );
train = intersect( liteIdx, find( imdb.ttSplit == 1 ) );
trainLab = imdb.clsLabel( train );

nClass  = 10;
nTrain  = length( train );
nTest   = length( test );

trainK = kernel( train, train );
testK  = kernel( test, train );

% tune regu paramter
selLambda = [ 0.00001 0.0001 0.001 ]
trainPred = zeros( nTrain, length( selLambda ) );
testPred = zeros( nTest, length( selLambda ) );
for s = 1 : length( selLambda )
  
  lambda = selLambda( s );
  fprintf( 'lambda = %f\n', lambda );

  funObj = @(u)SoftmaxLoss2( u, trainK, trainLab, nClass );
  fprintf('Training linear kernel multinomial logistic regression model...\n');
  uLinear = minFunc(@penalizedKernelL2_matrix,randn(nTrain*(nClass-1),1),options,trainK,nClass-1,funObj,lambda);
  uLinear = reshape( uLinear, [ nTrain nClass - 1 ] );
  uLinear = [ uLinear zeros( nTrain, 1 ) ];

  % Compute training errors
  [ ~, trainPred( :, s ) ] = max( trainK * uLinear, [], 2 );
  [ ~, testPred( :, s ) ] = max( testK * uLinear, [], 2 );
  fprintf( 'Train acc %.2f %%\n', 100 * sum( trainPred( :, s ) == trainLab ) / nTrain );
  fprintf( 'Test acc %.2f %%\n', 100 * sum( testPred( :, s ) == testLab ) / nTrain );
end % end for each lambda