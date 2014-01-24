function [ wSoftmax, probAll ] = MultiLRL2( xTrain, yTrain, xAll )
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
% display iteration info
options.Display = 1;


% Add bias
xTrain = [ ones( nTrain, 1 ) xTrain ];
xAll = [ ones( size( xAll, 1 ), 1 ) xAll ];

% softmax loss
funObj = @(W)SoftmaxLoss2( W, xTrain, yTrain, nClass );
% regularization paramters
lambda = 1 * ones( nVars + 1, nClass - 1 );
lambda( 1 , : ) = 0; % Don't penalize biases

PrintTab();fprintf( 'Training multinomial logistic regression model...\n' );
wSoftmax = minFunc(@penalizedL2,zeros((nVars+1)*(nClass-1),1),options,funObj,lambda(:));

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