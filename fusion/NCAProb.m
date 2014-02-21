function [ Anew, probAll ] = NCAProb( xTrain, yTrain, xAll )
%% NCAProb
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

% convert yTrian to indicator matrix
yInd = full( sparse( 1 : nTrain, yTrain, 1 ) );
A    = full( sparse( 1 : nClass, 1 : nVars, 1 ) );

% train NCA
% minFunc options
% options.Method = 'sd';
options.MaxIter = 500;
disp( options );

funObj = @(W)nca_obj( W, xTrain, yInd );
Anew = minFunc( funObj, A( : ), options );

Anew = reshape( Anew, nClass, nVars );

% get p_ij
p = zeros( nAll, nTrain );
allT   = ( Anew * xAll' )';
trainT = ( Anew * xTrain' );
for c = 1 : nClass
  tmp = bsxfun( @minus, allT( :, c ), trainT( c, : ) );
  p = p - tmp .* tmp;
end
p = bsxfun( @rdivide, exp( p ), sum( exp( p ), 2 ) );

% get prob all
probAll = zeros( nAll, nClass );
for c = 1 : nClass
  cIdx = find( yTrain == c );
  probAll( :, c ) = sum( p( :, cIdx ), 2 );
end

% end function NCAProb