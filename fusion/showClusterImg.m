imgID = [  8325 8352 8360 8375 8404 8432 8434 8435 8439 8443 8447 8448 8454 8533 8565 8596 8712 8830 8839   16289 16297 16312 16321 16346 16354 16357 16366 16380 16381 16391 16404 16410 16419 16420 ...
];
SIZE = ceil( sqrt( length( imgID ) ) );

for t = 1 : length( imgID )
  subplot( SIZE, SIZE, t ); 
  imagesc( imread( fullfile( imdb.imgDir, imdb.imgName{ imgID( t ) } ) ) );
  set(gca,'xtick',[],'ytick',[])
end