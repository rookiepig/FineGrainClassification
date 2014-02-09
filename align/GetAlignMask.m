%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: GetAlighMask
% Desc: get align mask for a input image with bounding box
% Author: Zhang Kang
% Date: 2013/11/30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ mask, abCnt ] = GetAlignMask( I, box )
%GETALIGNMASK Summary of this function goes here
%   Detailed explanation goes here

DEBUG  = false;
BIN_DIR = '..\..\bin\';
TEMP_IMG  = 'img.png';
TEMP_MASK = 'mask.png';
abCnt = 0;

if DEBUG
  % show image and bounding box
  figure;imshow( I );
  rectangle( 'Position', box, 'LineWidth',2, 'EdgeColor','b' );
end


% run grabcut.exe to segment image
imwrite( I, TEMP_IMG );

wid = size( I, 2 );
hei = size( I, 1 );
if( wid == box( 3 ) && hei == box( 4 ) )
  % bounding box is the whole image
  % narrow down 1 pixel
  fprintf( 1, 'Special -- Full Image Boudning Box\n' );
  abCnt = 1;
  box( 1 ) = box( 1 ) + 10;
  box( 2 ) = box( 2 ) + 10;
  box( 3 ) = box( 3 ) - 10;
  box( 4 ) = box( 4 ) - 10;
end

EXEXC_STR = sprintf( '%sgrabcut.exe %s %f %f %f %f %s', ...
  BIN_DIR, ...
  TEMP_IMG, box( 1 ), box( 2 ), box( 3 ), box( 4 ), TEMP_MASK );
fprintf( 1, '\tRun GrabCut\n' );
system( EXEXC_STR, '-echo' );

% load segment mask
mask = imread( TEMP_MASK );

if DEBUG
  figure;imshow( mask );
end

% get mask pixels
[ fgY, fgX ] = find( mask > 0 );

if( length( fgY ) <= 10 )
  % empty fore ground
  % set all bounding box pixels to fore ground
  fprintf( 1, 'Special - No Fore Ground Segment\n' );
  abCnt = 1;
  mask( box( 2 ) : box( 2 ) + box( 4 ), box( 1 ) : box( 1 ) + box( 3 ) ) = 128;
  [ fgY, fgX ] = find( mask > 0 );
end
fgCord = [ fgY fgX ];

% PCA to get eigen vector
meandCord = mean( fgCord );
[ coeff, score, latent ] = pca( fgCord );
e1 = coeff( :, 1 ) .* sqrt( latent( 1 ) );
e2 = coeff( :, 2 ) .* sqrt( latent( 2 ) );
e1 = e1';
e2 = e2';
if DEBUG
  % draw ellispe
  hold on;
  ang = atan( e1( 1 ) / e1( 2 ) );
  ellipse( 2 * norm( e1, 2 ), 2 * norm( e2, 2 ), ...
    ang , meandCord( 2 ), meandCord( 1 ), 'r' );
end
% get head and tail point
ratio = 3;
if( e1( 1 ) <= 0 )
  headPt = meandCord + ratio * e1;
  tailPt = meandCord - ratio * e1;
else
  headPt = meandCord - ratio * e1;
  tailPt = meandCord + ratio * e1;
end
if DEBUG
  % draw head, mean, tail point
  plot( meandCord( 2 ), meandCord( 1 ), 'r*' );
  plot( headPt( 2 ), headPt( 1 ), 'g*' );
  plot( tailPt( 2 ), tailPt( 1 ), 'b*' );
end

% give 4 labels to original mask
projCord = fgCord * coeff;
headPt   = headPt * coeff;
tailPt   = tailPt * coeff;
cordNum  = size( projCord, 1 );
for cordI = 1 : cordNum
  cordY = fgCord( cordI, 1 );
  cordX = fgCord( cordI, 2 );
  cordRatio = ( projCord( cordI, 1 ) - headPt( 1 ) ) / ( tailPt( 1 ) - headPt( 1 ) );
  if( cordRatio <= 0.25 )
    mask( cordY, cordX ) = 64;
  elseif ( cordRatio > 0.25 && cordRatio <= 0.5 )
    mask( cordY, cordX ) = 128;
  elseif ( cordRatio > 0.5  && cordRatio <= 0.75 )
    mask( cordY, cordX ) = 192;
  elseif ( cordRatio > 0.75 )
    mask( cordY, cordX ) = 255;
  end
end
if DEBUG
  % show new mask
  figure; imshow( mask );
end

