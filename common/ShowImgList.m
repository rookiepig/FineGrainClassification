function ShowImgList( imgList )
%% ShowImgList
%  Desc: show a sery of image
%  In: 
%    imgList -- cell array of image images
%  Out:
%    
%%

imgNum = numel( imgList );
GRID_SZ = ceil( sqrt( imgNum ) );

figure;
for t = 1 : imgNum
  % subplot and show image in each grid
  subplot( GRID_SZ, GRID_SZ, t );
  imshow( imread( imgList{ t } ) );
end
% end ShowImgList