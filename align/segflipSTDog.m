%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: segflipSTDog 
% Desc: segment Stanford Dog dataset
% Author: Zhang Kang
% Date: 2013/12/23
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
DOG_DIR = '../../in/STDog/';
load( fullfile( DOG_DIR, 'file_list.mat' ) );


imgNum = numel(  file_list );
dogBdbox = zeros( imgNum, 4 );

totalAb = 0;

% for imgI = 8457 : imgNum
%   fprintf( 1, 'Img: %d\n', imgI );
%   imgFn = sprintf( '%sImages/%s', DOG_DIR, file_list{ imgI } );
%   anoFn = sprintf( '%sAnnotation/%s', DOG_DIR, annotation_list{ imgI } );
%   k = strfind( imgFn, '.jpg' );
%   maskFn = sprintf( '%s_m.png', imgFn( 1 : k ) );
%   flipFn = sprintf( '%s_f.png', imgFn( 1 : k ) );
%   flipMaskFn = sprintf( '%s_f_m.png', imgFn( 1 : k ) );
%   fprintf( 1, '\n\toriginal mask' );
%   img   = imread( imgFn );
%   box = GetDogBdbox( anoFn );
%   [ mask, abCnt ]  = GetAlignMask( img, box );
%   % record total abnormal num
%   totalAb = totalAb + abCnt;
%   imwrite( mask, maskFn );
%   
%   % get flip mask
%   fprintf( 1, '\n\tflip mask' );
%   flipImg = img;
%   for c = 1 : size( img, 3 )
%     flipImg( :, :, c ) = fliplr( img( :, :, c ) );
%   end
%   imwrite( flipImg, flipFn );
%   flipBox = box;
%   flipBox( 1 ) = size( img, 2 ) - ( box( 1 ) + box( 3 ) );
%   [ mask, abCnt ]  = GetAlignMask( flipImg, flipBox );
%   totalAb = totalAb + abCnt;
%   imwrite( mask, flipMaskFn );
% end

imgSize = zeros( imgNum, 2 );
for imgI = 1 : imgNum
    fprintf( 1, 'Img: %d\n', imgI );
    imgFn = sprintf( '%simages/%s', DOG_DIR, file_list{ imgI } );
    img   = imread( imgFn );
    wid = size( img, 2 );
    hei = size( img, 1 );
    imgSize( imgI, : ) = [ wid, hei ];
end

fprintf( '\nTotal Abnormal Number: %d (%%%.2f)', ...
  totalAb, 100 * totalAb / imgNum );
