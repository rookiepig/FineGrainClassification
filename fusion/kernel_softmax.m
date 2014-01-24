%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: kernel_softmax.m
% Desc: kernel multinomial l2 regularizatioin
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

test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );

nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
nTrain  = length( train );
nTest   = length( test );

trainK = kernel( train, train );
testK  = kernel( test, train );

% regu paramter
lambda = 1e-2;


funObj = @(u)SoftmaxLoss2( u, trainK, trainLab, nClass );
fprintf('Training linear kernel multinomial logistic regression model...\n');
uLinear = minFunc(@penalizedKernelL2_matrix,randn(nTrain*(nClass-1),1),options,trainK,nClass-1,funObj,lambda);
uLinear = reshape( uLinear, [ nTrain nClass - 1 ] );
uLinear = [ uLinear zeros( nTrain, 1 ) ];


% Compute training errors
[ ~, trainPred ] = max( trainK * uLinear, [], 2 );

[ ~, testPred ] = max( testK * uLinear, [], 2 );

fprintf( 'Train acc %.2f %%\n', 100 * sum( trainPred == trainLab ) / nTrain );
fprintf( 'Test acc %.2f %%\n', 100 * sum( testPred == testLab ) / nTrain );