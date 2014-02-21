function [ Anew, probAll ] = NCAProb_knn( xTrain, yTrain, xAll, regLambda )
%% NCAProb_knn
%  Desc: NCA to get probability output
%  In: 
%    xTrain -- (nTrain * nVars) training sample feature
%    yTrain -- (nTrain * 1) training sample label
%    xAll   -- (nAll * nVars) all sample feature
%  Out:
%    Anew -- (nClass * nVars) NCA transformation matrix
%    probAll   -- nca probability
%%

PrintTab();fprintf( 'function: %s\n', mfilename );

nTrain = size( xTrain, 1 );
nClass = max( yTrain );
nVars = size( xTrain, 2 );
nAll  = size( xAll, 1 );
PrintTab(); fprintf( 'nTrain %d - nVars %d - nAll %d - nClass %d\n', nTrain, nVars, nAll, nClass );
PrintTab(); fprintf( 'regLambda: %f\n', regLambda );

% convert yTrian to indicator matrix
yInd = full( sparse( 1 : nTrain, yTrain, 1 ) );
A    = full( sparse( 1 : nClass, 1 : nVars, 1 ) );

% knn for training and all
K = 60;
PrintTab(); fprintf( 'knn K = %d', K );
trainIdx = knnsearch( xTrain, xTrain, 'K', K );
allIdx   = knnsearch( xTrain, xAll, 'K',  K );
% trainIdx = ones( nTrain, K );
% allIdx   = ones( nAll, K );

% train NCA
% minFunc options
options.MaxIter = 500;
disp( options );

% save temp results
if( ~exist( 'tmp/Anew.mat', 'file' ) )
  % minimize nca to get Anew
  funObj = @(W)nca_obj_knn( W, xTrain, yInd, trainIdx' );
  % regularization paramter
  lambda = regLambda * ones( size( A ) );
  Anew = minFunc( @penalizedL2, A( : ), options, funObj, lambda( : ) );
  Anew = reshape( Anew, nClass, nVars );
  save( 'tmp/Anew.mat', 'Anew' );
else
  fprintf( 'load Anew from Anew.mat\n' );
  load( 'tmp/Anew.mat' );
end

% get p_ij
p = zeros( nAll, nTrain );
allT   = ( Anew * xAll' )';
trainT = ( Anew * xTrain' )';
for n = 1 : nAll
  nIdx = allIdx( n, : );
  for t = 1 : K
    j = nIdx( t );
    tmp = allT( n, : ) - trainT( j, : );
    p( n, j ) =  exp( - tmp * tmp' );
  end % end for K nn
end % end for nAll
p = bsxfun( @rdivide, p, sum( p, 2 ) );

% get prob all
probAll = zeros( nAll, nClass );
for c = 1 : nClass
  cIdx = find( yTrain == c );
  probAll( :, c ) = sum( p( :, cIdx ), 2 );
end

% end function NCAProb_knn