%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: setupSTDog.m
% Desc: setup Stanford Dog dataset and store files
% Author: Zhang Kang
% Date: 2013/12/23
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 1, '\n Setup Stanford Dog dataset ... \n' );
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
  imdb.imgDir = '../../in/STDog/Images';
  load( fullfile( '../../in/STDog/', 'file_list.mat' ) );
  load( fullfile( '../../in/STDog/', 'imgSize.mat' ) );
  % remove the ID column of each data
  clsLabel = labels;
  clsName  = cell( 120, 1 );
  imgName  = file_list;

  % prepare bdBox and clsName
  imgNum = numel(  file_list );
  bdBox  = zeros( imgNum, 4 );
  DOG_DIR = '../../in/STDog/';
  for imgI = 1 : imgNum
    fprintf( '\t Img: %d - process bdBox and clsName\n', imgI );
    % bdBox
    anoFn = sprintf( '%sAnnotation/%s', DOG_DIR, annotation_list{ imgI } );
    box = GetDogBdbox( anoFn );
    bdBox( imgI, : ) = box;
    % clsName
    kSt = strfind( file_list{ imgI }, '-' );
    kEd = strfind( file_list{ imgI }, '/' );
    curClsName = sprintf( '%s', file_list{ imgI }( kSt + 1 : kEd - 1 ) );
    clsName{ clsLabel( imgI ) } = curClsName;
  end
  % prepare train&test split
  ttSplit =  false( imgNum, 1 ) ;
  load( fullfile( '../../in/STDog/', 'train_list.mat' ) );
  % now file_list are training file name
  for imgI = 1 : imgNum  
    curImgName = imgName{ imgI };
    isTrain = strcmp( curImgName, file_list );
    if( sum( isTrain ) )
      ttSplit( imgI ) = 1;
    else
      ttSplit( imgI ) = 0;
    end
  end

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
    ttSplit  = [ ttSplit; ttSplit( selTrain ) ];
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