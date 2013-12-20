%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: segflipCUB11
% Desc: segment CUB 2011 dataset
% Author: Zhang Kang
% Date: 2013/11/30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CUB_DIR = '../../in/CUB2011/';
load( 'boxes.mat' );
load( 'images.mat' );

imgNum = size( boxes, 1 );
totalAb = 0;


for imgI = 1 : imgNum
  fprintf( 1, 'Img: %d\n', imgI );
  imgFn = sprintf( '%simages/%s', CUB_DIR, images{ imgI, 2 } );
  k = strfind( imgFn, '.jpg' );
  maskFn = sprintf( '%s_m.png', imgFn( 1 : k ) );
  flipFn = sprintf( '%s_f.png', imgFn( 1 : k ) );
  flipMaskFn = sprintf( '%s_f_m.png', imgFn( 1 : k ) );
  fprintf( 1, '\n\toriginal mask' );
  img   = imread( imgFn );
  box   = boxes( imgI, 2 : 5 );
  [ mask, abCnt ]  = GetAlignMask( img, box );
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
