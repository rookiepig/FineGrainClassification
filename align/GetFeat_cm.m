%% GetFeat_cm: get color moment features (can use mask) 
% -----------------------------------------------------------------
function [ features ] = GetFeat_cm( img, varargin )
% -----------------------------------------------------------------

% declare global variables
global conf imdb;

% be careful about return sequence
[ descr, frame ] = local_fea_extraction( img, conf.cmParam );

if( nargin == 2 )
  % mask out unusable descriptors
  mask = varargin{ 1 };
  [ maskY, maskX ] = find( mask > 0 );
  tmpFrame = transpose( round( frame ) );         % round float keypoint
  maskCord = [ maskX maskY ];
  ptIdx = ismember( tmpFrame( :, 1 : 2 ), maskCord, 'rows' );  % get kyepoint index
  descr = descr( :,  ptIdx  );
  frame = frame( :, ptIdx );
end


if( conf.removeZeroFeat )
  % kick out zero descriptor
  absDesc   = sum( abs( descr ) );
  nonZero   = find( absDesc > 1e-10 );
  if( ~isempty( nonZero ) )
    features.frame = frame( :, nonZero );
    features.descr = descr( :, nonZero );
  else
    fprintf( 'Warning: non-zero SIFT equals 0\n' );
    features.frame = frame;
    features.descr = descr;
  end
else
  features.frame = frame;
  features.descr = descr;
end
