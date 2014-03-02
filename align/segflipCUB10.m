%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: segflipCUB10
% Desc: segment CUB 2010 dataset
% Author: Zhang Kang
% Date: 2014/02/23
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CUB_DIR = '../../in/CUB10/';
IMG_DIR = '../../in/CUB10/images/';
ANO_DIR = '../../in/CUB10/annotations-mat/';
load( fullfile( CUB_DIR, 'cub2010.mat' ) );

imgNum = length( ttSplit );
% imgNum = 1;
segAcc = zeros( imgNum, 1 );

totalAb = 0;

for imgI = 1 : imgNum
  fprintf( 1, 'Img: %d\n', imgI );
  % set image names
  imgFn = fullfile( IMG_DIR, imgName{ imgI } );
  k = strfind( imgFn, '.jpg' );
  maskFn = sprintf( '%s_m.png', imgFn( 1 : k - 1 ) );
  flipFn = sprintf( '%s_f.png', imgFn( 1 : k - 1) );
  flipMaskFn = sprintf( '%s_f_m.png', imgFn( 1 : k - 1 ) );
  fprintf( 1, '\n\toriginal mask' );
  img   = imread( imgFn );
  % get bounding box & ground truth segment
  tmpFn = fullfile( ANO_DIR, imgName{ imgI } );
  k = strfind( tmpFn, '.jpg' );
  anoFn = sprintf( '%s.mat', tmpFn( 1 : k - 1 ) );
  load( anoFn );
  box   = [ bbox.left, bbox.top, bbox.right - bbox.left, bbox.bottom - bbox.top ];
  [ mask, abCnt ]  = GetAlignMask( img, box );
  % get segment acc
  m = ( mask > 0 );
  segAcc = 100 * sum( m( : ) & seg( : ) ) / sum( m( : ) | seg( : ) );
  clear( 'bbbox', 'seg' );
  % record total abnormal num
  totalAb = totalAb + abCnt;
  imwrite( mask, maskFn );
  
  % get flip mask
  fprintf( 1, '\n\tflip mask' );
  flipImg = img;
  for c = 1 : size( img, 3 )
    flipImg( :, :, c ) = fliplr( img( :, :, c ) );
  end
  imwrite( flipImg, flipFn );
  flipBox = box;
  flipBox( 1 ) = size( img, 2 ) - ( box( 1 ) + box( 3 ) );
  [ mask, abCnt ]  = GetAlignMask( flipImg, flipBox );
  totalAb = totalAb + abCnt;
  imwrite( mask, flipMaskFn );
end

% imgSize = zeros( imgNum, 2 );
% for imgI = 1 : imgNum
%     fprintf( 1, 'Img: %d\n', imgI );
%     imgFn = sprintf( '%simages/%s', CUB_DIR, images{ imgI, 2 } );
%     img   = imread( imgFn );
%     wid = size( img, 2 );
%     hei = size( img, 1 );
%     imgSize( imgI, : ) = [ wid, hei ];
% end

fprintf( '\nTotal Abnormal Number: %d (%%%.2f)', ...
  totalAb, 100 * totalAb / imgNum );

fprintf( '\nSegment Acc: %.2f %%', ...
  mean( segAcc ) );