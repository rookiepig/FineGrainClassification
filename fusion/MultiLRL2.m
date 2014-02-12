function [ wSoftmax, probAll ] = MultiLRL2( xTrain, yTrain, xAll, regLambda, bias )
%% MultiLRL2
%  Desc: Multinomial logistic regression with L2-regularization
%        **automatically add bias term to xTrain and xAll**
%  In: 
%    xTrain -- (nTrain * nVars) training sample feature
%    yTrain -- (nTrain * 1) training sample label
%    xAll   -- (nAll * 1) all sample feature
%  Out:
%    wSoftmax -- (nVars + 1 * nClass) calculated softmax weights
%    probAll   -- normalized probability
%%

PrintTab();fprintf( 'function: %s\n', mfilename );

nTrain = size( xTrain, 1 );
nClass = max( yTrain );
nVars = size( xTrain, 2 );

% minFunc options
% options.Method = 'sd';
options.MaxIter = 1000;
disp( options );

% Add bias
xTrain = [ ones( nTrain, 1 ) xTrain ];
xAll = [ ones( size( xAll, 1 ), 1 ) xAll ];

% softmax loss
% funObj = @(W)WeightSoftmaxLoss2( W, xTrain, yTrain, nClass, bias );    % weighted softmax
funObj = @(W)SoftmaxLoss2( W, xTrain, yTrain, nClass );

% weight initialization
tmp     = zeros( nVars + 1, nClass - 1 );
% set to unit matrix
tmp( 1, : ) = 0;
e = eye( nClass );
tmp( 2 : end, : ) = e( :, 1 : end - 1 );
w0 = tmp( : );
% regularization paramters
lambda = regLambda * ones( nVars + 1, nClass - 1 );
lambda( 1 , : ) = 0; % Don't penalize biases

PrintTab();fprintf( 'Training multinomial logistic regression model...\n' );
wSoftmax = minFunc( @penalizedL2, w0, options, funObj, lambda( : ) );

% reshape softmax paramters
wSoftmax = reshape( wSoftmax, [ nVars + 1 nClass - 1 ] );
wSoftmax = [ wSoftmax zeros( nVars + 1, 1 ) ];

% get training accuracy
[ ~, yhat ] = max( xTrain * wSoftmax, [], 2 );
trainAcc = 100 * sum( yhat == yTrain ) / length( yTrain );
PrintTab();fprintf( 'MultiLRL2 train acc %.2f %%\n', trainAcc );

% get all sample probability
p = exp( xAll * wSoftmax );
Z = sum( p, 2 );
probAll = p ./ repmat( Z, [ 1, nClass ] );

% end function MultiLRL2