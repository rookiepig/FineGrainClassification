% try sample clustering

% get configuration
conf = InitConf( );
% load imdb, kernel
load( conf.imdbPath );

% load kernel
if( ~exist( 'kernel', 'var' ) )
  PrintTab();fprintf( 'loading kernel (maybe slow)\n' );
  load( conf.kernelPath );
  for y = 1 : size( kernel, 1 )
    kernel( y, y ) = 0;
  end
else
  PrintTab();fprintf( 'kernel exist\n' );
end

% recursive split sample
nSample = length( imdb.clsLabel );
train = find( imdb.ttSplit == 1 );
test  = find( imdb.ttSplit == 0 );


% spectral clustering
CLUSTER_NUM = 200;
cluster = cell( 1, CLUSTER_NUM );
[ C, ~, ~ ] = SpectralClustering( kernel, CLUSTER_NUM, 3 );
% get left and right kernel and sample label
for  k = 1 : CLUSTER_NUM
  cluster{ k } = find( C == k );
end


% clear split;
% clear splitSVM;
% clear MAX_DEPTH;

% global split;
% global splitSVM;
% global kernel;
% global MAX_DEPTH;
% global negKernel;



% MAX_DEPTH = 10;
% BinarySplitSample( 1 : nSample, 1, 1 );


% acc = zeros( nSample );
% % calculate test accuracy
% for t = 1 : numel( split )
%   if( ~isempty( split{ t } ) )
%     if( split{ t }.isLeaf )
%       % only handle leaf node
%       fprintf( 'Leaf node %d -- size %d\n', t, length( split{ t }.sampleLab ) );
%       % train always right
%       curTrain = intersect( train, split{ t }.sampleLab );
%       acc( curTrain ) = 1;
%       curTrainLab = imdb.clsLabel( curTrain );
%       split{ t }.trainCls = find( histc( curTrainLab, 1 : 200 ) > 0 );
%       % test accuracy
%       curTest  = intersect( test, split{ t }.sampleLab );
%       curTestLab = imdb.clsLabel( curTest );
%       split{ t }.testCls = find( histc( curTestLab, 1 : 200 ) > 0 );
%       acc( curTest ) = ismember( curTestLab, curTrainLab );
%     end
%   end
% end % for each node

% trainAcc = sum( acc( train ) ) / length( train )
% testAcc  = sum( acc( test ) ) / length( test )


% get test node index
% testNode = ones( length( test ), MAX_DEPTH );
% testAcc  = ones( length( test ), MAX_DEPTH );

% for t = 1 : length( test )
%   for d = 2 : MAX_DEPTH
%     nodeIdx = testNode( t, d - 1 );
%     curSVM = splitSVM{ testNode( t, d - 1 ) };
%     testK = kernel( test( t ), split{ nodeIdx }.sampleLab );
%     testK = [ ( 1 : size( testK, 1 ) )', testK ];
%     [ pred, acc, sc ] = libsvmpredict( 1, double( testK ), curSVM  );
%     if( pred == 1 )
%       testNode( t, d ) = 2 * nodeIdx;
%     elseif( pred == 2 )
%       testNode( t, d ) = 2 * nodeIdx + 1;
%     end
%     % record accuracy
%     nodeLab = imdb.clsLabel( split{ testNode( t, d ) }.sampleLab );
%     % h = histc( nodeLab, 1 : 200 );
%     testAcc( t, d ) = ismember( imdb.clsLabel( test( t ) ),  nodeLab );
%   end % end for each depth
% end % end for each sample

% for d = 1 : MAX_DEPTH
%   fprintf( 'Depth %d: ', d );
%   fprintf( 'Acc: %.2f %%\n', 100 * sum( testAcc( :, d ) ) / length( test ) );
% end
% % end function GroupClustering



