%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: setupCUB11.m
% Desc: setup CUB 2011 dataset and store file
% Author: Zhang Kang
% Date: 2013/12/05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 1, '\n Setup CUB 2011 dataset ... \n' );
% declare global variables
global conf imdb;

if ~exist( conf.imdbPath, 'file' )
  % datset mat file does not exit
  % imdb structure without left-right flip ( M samples N classes )
  % -- imgDir  : root directory for dataset
  % -- bdBox    : M * 4 matrix, each row [ x y wid hei ] bounding box
  % -- clsLabel : M * 1 column vector, class label for each image
  % -- clsName  : N * 1 cell vector, class name for each label
  % -- imgName  : M * 1 cell vector, image file name
  % -- imgSize  : M * 2 matrix, image size [ wid hei ]
  % -- ttSplit  : M * 1 0-1 vector, 1 means training
  imdb.imgDir = '../../in/CUB2011/images';
  load( fullfile( '../../in/CUB2011/', 'cub2011.mat' ) );
  % remove the ID column of each data
  bdBox = bdBox( :, 2 : 5 );
  clsLabel = clsLabel( :, 2 );
  clsName = clsName( :, 2 );
  imgName = imgName( :, 2 );
  ttSplit = ttSplit( :, 2 );
  if( conf.useSegMask )
    % add mask file name
    maskName = cell( numel( imgName ), 1 );
    for imIdx = 1 : numel( imgName )
      imgFn = imgName{ imIdx };
      k = strfind( imgFn, '.jpg' );
      maskFn = sprintf( '%s_m.png', imgFn( 1 : k ) );
      maskName{ imIdx } = maskFn;
    end
    
  end
  if( conf.isLRFlip )
    % enable left right flip
    % enlarge training data, test data remains unchanged
    selTrain = ( find( ttSplit == 1 ) );
    ttSplit = [ ttSplit; ttSplit( selTrain ) ];
    clsLabel = [ clsLabel; clsLabel( selTrain ) ];
    
    newImgName = cell( numel( selTrain ), 1 );
    newBdBox = zeros( numel( selTrain ), 4 );
    if( conf.useSegMask )
      newMaskName = cell( numel( selTrain ), 1 );
    end
    for ii = 1 : numel( selTrain )
      imgFn = imgName{ selTrain( ii ) };
      curBox = bdBox( selTrain( ii ), : );
      k = strfind( imgFn, '.jpg' );
      flipFn = sprintf( '%s_f.png', imgFn( 1 : k ) );
      newImgName{ ii } = flipFn;
      flipBox = curBox;
      flipBox( 1 ) = imgSize( selTrain( ii ), 1 ) - ...
        ( curBox( 1 ) + curBox( 3 ) );
      newBdBox( ii, : ) = flipBox;
      if( conf.useSegMask )
        flipMaskFn = sprintf( '%s_f_m.png', imgFn( 1 : k ) );
        newMaskName{ ii } = flipMaskFn;
      end
    end
    imgName = [ imgName; newImgName ];
    bdBox = [ bdBox; newBdBox ];
    if( conf.useSegMask )
      maskName = [ maskName; newMaskName ];
    end
  end
  
  imdb.bdBox = bdBox;
  imdb.clsLabel = clsLabel;
  imdb.clsName = clsName;
  imdb.imgName = imgName;
  imdb.ttSplit = ttSplit;
  if( conf.useSegMask )
    imdb.maskName = maskName;
  end
  
  % handle lite version (preserve first 20 class)
  if( conf.lite )
    imdb.clsName = clsName( 1 : 5 );
    for c = 1 : 5
      sel{ c } = find( imdb.clsLabel == c );
    end
    sel = cat( 1, sel{ : } );
    imdb.bdBox = bdBox( sel, : );
    imdb.clsLabel = clsLabel( sel );
    imdb.imgName = imgName( sel );
    imdb.ttSplit = ttSplit( sel );
    if( conf.useSegMask )
      imdb.maskName = maskName( sel );
    end
  end
  % save imdb var
  save( conf.imdbPath, 'imdb' );
else
  load( conf.imdbPath );
end

fprintf( 1, '\n ... Done\n' );