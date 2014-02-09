function [ wSoftmax, probAll ] = MultiLRL2_all( xTrain, yTrain, trainPrior, xAll,  allPrior,  clsToCluster, regLambda )
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

% minFunc options
% options.Method = 'sd';
options.MaxIter = 1000;
disp( options );

nCluster = size( clsToCluster, 2 );

for t = 1 : nCluster
  % set each cluster constant variable
  nClass{ t }   = sum( clsToCluster( :, t ) ); 
  nTrain{ t } = size( xTrain{ t }, 1 );
  nVars{ t } = size( xTrain{ t }, 2 );
  % Add bias
  xTrain{ t } = [ ones( nTrain{ t }, 1 ) xTrain{ t } ];
  xAll{ t } = [ ones( size( xAll{ t }, 1 ), 1 ) xAll{ t } ];
end

% softmax loss
funObj = @(W)SoftmaxLoss2_all( W, xTrain, yTrain, nClass, trainPrior, clsToCluster );    % weighted softmax
% funObj = @(W)SoftmaxLoss2( W, xTrain, yTrain, nClass );

% initial weight and regularization paramters
for t = 1 : nCluster
  w0{ t } = zeros( ( nVars{ t } + 1 ) * ( nClass{ t } - 1 ) , 1 );
  lambda{ t } = regLambda * ones( nVars{ t } + 1, nClass{ t } - 1 );
  lambda{ t }( 1 , : ) = 0; % Don't penalize biases
  lambda{ t } = reshape( lambda{ t }, [ ( nVars{ t } + 1 ) * ( nClass{ t } - 1 ), 1 ] );
end
w0 = cat( 1, w0{ : } );
lambda = cat( 1, lambda{ : } );

PrintTab();fprintf( 'Training multinomial logistic regression model...\n' );
wSoftmax = minFunc( @penalizedL2, w0, options, funObj, lambda );

% reshape softmax paramters
% wSoftmax = reshape( wSoftmax, [ nVars + 1 nClass - 1 ] );
% wSoftmax = [ wSoftmax zeros( nVars + 1, 1 ) ];

wLoc = 1;
wIdx = cell( 1, nCluster );
w = cell( 1, nCluster );
for t = 1 : nCluster
    [ n{ t }, p{ t } ] = size( xTrain{ t } );
    w{ t } = wSoftmax( wLoc : ( wLoc + ( nVars{ t } + 1 ) * ( nClass{ t } - 1 ) ) - 1 );
    w{ t } = reshape( w{ t }, [ nVars{ t } + 1, nClass{ t } - 1 ] );
    w{ t }( :, nClass{ t } ) = zeros( nVars{ t } + 1, 1 );    % avoid reduntant weight
    zTrain{ t } = sum( exp( xTrain{ t } * w{ t } ), 2 );
    zAll{ t } = sum( exp( xAll{ t } * w{ t } ), 2 );
    wIdx{ t } = wLoc;
    wLoc = wLoc + p{ t } * ( nClass{ t } - 1 );
end

% class probability for each sample
trainSample = size( xTrain{ 1 }, 1 );
allClass  = max( yTrain );
probTrain = zeros( trainSample, allClass );
for t = 1 : nCluster
    clsIdx = find( clsToCluster( :, t ) == 1 );
    probTrain( :, clsIdx ) = probTrain( :, clsIdx ) + ...
      exp( xTrain{ t } * w{ t } ) .* repmat( trainPrior( :, t ), [ 1, nClass{ t } ] ) ./ repmat( zTrain{ t }, [ 1, nClass{ t } ] );
end

allSample = size( xAll{ 1 }, 1 );
allClass  = max( yTrain );
probAll = zeros( allSample, allClass );
for t = 1 : nCluster
    clsIdx = find( clsToCluster( :, t ) == 1 );
    probAll( :, clsIdx ) = probAll( :, clsIdx ) + ...
      exp( xAll{ t } * w{ t } ) .* repmat( allPrior( :, t ), [ 1, nClass{ t } ] ) ./ repmat( zAll{ t }, [ 1, nClass{ t } ] );
end

% get training accuracy
[ ~, yhat ] = max( probTrain, [], 2 );
trainAcc = 100 * sum( yhat == yTrain ) / length( yTrain );
PrintTab();fprintf( 'MultiLRL2 train acc %.2f %%\n', trainAcc );


% end function MultiLRL2