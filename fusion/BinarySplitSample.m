function BinarySplitSample( sampleLab, depth, nodeIdx )
%% BinarySplitSample
%  Desc: binary split sample
%  In: 
%    kernel - (nSample * nSample) kernel matrix
%    sampleLab - (nSample * 1) label vector
%    depth - cluster depth
%  Out:
%  
%%

global split;
global splitSVM;
global kernel;
global MAX_DEPTH;

PrintTab();fprintf( 'depth: %d\n', depth );
% PrintTab();fprintf( '%d,',sampleLab );
split{ nodeIdx }.sampleLab = sampleLab;

if( depth <= MAX_DEPTH && length( sampleLab ) > 50 )
  split{ nodeIdx }.isLeaf = 0;
  % spectral clustering
  [ C, ~, ~ ] = SpectralClustering( kernel( sampleLab, sampleLab ), 2, 3 );
  % get left and right kernel and sample label
  for  k = 1 : 2
    cluster{ k } = find( C == k );
  end

  % get left right label
  leftLab = sampleLab( cluster{ 1 } );
  % leftKernel = kernel( cluster{ 1 }, cluster{ 1 } );

  rightLab = sampleLab( cluster{ 2 } );
  % rightKernel = kernel( cluster{ 2 }, cluster{ 2 } );

  % trian split SVM
  % trainK = kernel( sampleLab, sampleLab );
  % trainK = [ ( 1 : size( trainK, 1 ) )', trainK ];
  % splitSVM{ nodeIdx } = libsvmtrain( double( C ), double( trainK ), '-c 10 -t 4 -q' );

  % recursive split left and right samples
  BinarySplitSample( leftLab, depth + 1, 2 * nodeIdx );
  BinarySplitSample( rightLab, depth + 1, 2 * nodeIdx + 1 );
else
  split{ nodeIdx }.isLeaf = 1;
end

% end BinarySplitSample