%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step1_trainEncoder.m
% Desc: get encoder and save results
% Author: Zhang Kang
% Date: 2013/12/08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step1: train encoder
clear;tic;
fprintf( '\n Step1: Train Encoder ...\n' );

% initial all configuration
initConf;

% setup dataset
switch conf.dataset
  case {'CUB11'}
    setupCUB11;
  case {'STDog'}
    setupSTDog;
end

if exist( conf.encoderPath, 'file' )
  % load existing encoder
  fprintf( '\n No need to train encoder (file: %s)\n', conf.encoderPath );
else
  nClass = max( imdb.clsLabel );
  encoder = cell( 1, nClass );
  % for each class train a encoder
  for c = 1 : nClass
    fprintf( '\t train encoder for class %d\n', c );
    encSelTrain = intersect( find( imdb.ttSplit == 1 ), find( imdb.clsLabel == c ) );
    if( conf.useSegMask )
      % use boudingbox + segment mask
      encoder{ c } = TrainEncoder( ...
        fullfile( imdb.imgDir, imdb.imgName( encSelTrain ) ), ...
        imdb.bdBox( encSelTrain, : ), ...
        fullfile( imdb.imgDir, imdb.maskName( encSelTrain ) ) ...
        );
    elseif( conf.useBoundingBox )
      % only use bounding box
      encoder{ c } = TrainEncoder( ...
        fullfile( imdb.imgDir, imdb.imgName( encSelTrain ) ), ...
        imdb.bdBox( encSelTrain, : ) ) ;
    end
  end % end for each class
  % save encoder
  save( conf.encoderPath, 'encoder' ) ;
end

% record time
fprintf( '\n ... Done Train Encoder time: %.2f (s)\n', toc );
