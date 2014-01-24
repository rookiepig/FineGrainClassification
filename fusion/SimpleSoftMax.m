function [ prob ] = SimpleSoftMax( score )
%% SimpleSoftMax
%  Desc: simple softmax to convert score to prob
%        x = x - max( x )  p( x ) = exp( x ) / sum( exp( x ) )
%  In: 
%    score -- (nSample * nclass) score
%  Out:
%    prob -- (nSample * nClass) prob
%%
% PrintTab();fprintf( 'function: %s\n', mfilename );

nSample = size( score, 1 );
nClass  = size( score, 2 );

% % l2 norm score
for m = 1 : nSample
  s = score( m, : );
  score( m, : ) = s ./ max( norm( s, 2 ), 1e-12 );
end % end for each sample

% get max score
maxScore = max( score, [], 2 );
maxScore = repmat( maxScore, [ 1, nClass ] );

% exp
score = exp( score );
% prob normalize
Z = sum( score, 2 );
Z = repmat( Z, [ 1, nClass ] );

prob = score ./ Z;

% end function SimpleSoftMax
