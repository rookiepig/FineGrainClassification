%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: kernelToSim.m
% Desc: convert kernel matrix to similarity matrix
% Author: Zhang Kang
% Date: 2014/01/02
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% init configuration
initConf;

if( exist( conf.clsSimPath ), 'file' )
  fprintf( 'Load class similarity matrix from %s\n', conf.clsSimPath );
  load( conf.clsSimPath );
else
  % load kernel
  load( conf.kernelPath );
  load( conf.imdbPath );

  train = find( imdb.ttSplit == 1 );
  numClasses = numel( imdb.clsName );
  clsSim     = zeros( numClasses, numClasses );

  % get class similarity
  for y = 1 : numClasses
    yIdx = intersect( find( imdb.clsLabel == y ), train );
    for x = 1 : numClasses
      xIdx = intersect( find( imdb.clsLabel == x ), train );
      tmpK = kernel( yIdx, xIdx );
      clsSim( y, x ) = sum( tmpK( : ) ) / ( length( yIdx ) * length( xIdx ) );
    end
  end

  save( )
end