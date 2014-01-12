function [clsSim] = KernelToSim( kernel, sampleLab, ttSplit )
%% KernelToSim
%  Desc: convert kernel matrix to class similarity matrix
%  In: 
%    kernel -- (nSample * nSample) kernel matrix
%    sampleLab -- (nSample * 1) sample class label
%    ttSplit -- (nSample * 1) sample training&testing split
%  Out:
%    clsSim -- (nClass * nClass) class similarity matrix
%%

PrintTab();fprintf( 'function: %s\n', mfilename );

train  = find( ttSplit == 1 );
nClass = max( sampleLab );
clsSim = zeros( nClass, nClass );

% get class similarity
for y = 1 : nClass
  yIdx = intersect( find( sampleLab == y ), train );
  for x = 1 : nClass
    xIdx = intersect( find( sampleLab == x ), train );
    clsSim( y, x ) = sum( sum( kernel( yIdx, xIdx ) ) ) / ...
      ( length( yIdx ) * length( xIdx ) );
  end
  if( y == x )
    clsSim( y, x ) = 0;
  end
end

% end function KernelToSim
