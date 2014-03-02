%% ReadImg: read image and add additional processing
% -----------------------------------------------------------------
function [ img ] = ReadImg( imgFn, varargin )
% -----------------------------------------------------------------

% declare global variables
global conf imdb;

% no bounding box
img = imread( imgFn );

if ( nargin == 2 )
  % crop image using bounding box
  curBox = varargin{ 1 };
  img = imcrop( img, curBox );
end

img = im2single( img );

if( conf.isStandImg )
  % standarize image --> max pixel < conf.maxPixNum
  wid = size( img, 2 );
  hei = size( img, 1 );
  
  % if( hei * wid > conf.maxPixNum )
  %   % resize image only when pixel number exceed
  %   scale = sqrt( conf.maxPixNum / ( hei * wid ) );
  %   % must use nearest neighbour to handle seg mask
  %   img = imresize( img, scale, 'nearest' );
  % end

  if( hei > wid )
    % must use nearest neighbour to handle seg mask
    img = imresize( img, [ conf.maxImgSz, NaN ], 'nearest' );
  else
    % must use nearest neighbour to handle seg mask
    img = imresize( img, [ NaN, conf.maxImgSz ], 'nearest' );
  end 
end
