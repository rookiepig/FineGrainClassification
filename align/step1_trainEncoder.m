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
setupCUB11;

if exist( conf.encoderPath, 'file' )
  % load existing encoder
  fprintf( '\n No need to train encoder (file: %s)\n', conf.encoderPath );
else
  numTrain = 5000 ;    % training images for encoding
  if( conf.lite )
    % lite version only use 100 images
    numTrain = 100;
  end
  encSelTrain = vl_colsubset( transpose( find( imdb.ttSplit == 1 ) ), ...
    numTrain, 'uniform' ) ;
  if( conf.useSegMask )
    % use boudingbox + segment mask
    encoder = TrainEncoder( ...
      fullfile( imdb.imgDir, imdb.imgName( encSelTrain ) ), ...
      imdb.bdBox( encSelTrain, : ), ...
      fullfile( imdb.imgDir, imdb.maskName( encSelTrain ) ) ...
      );
  elseif( conf.useBoundingBox )
    % only use bounding box
    encoder = TrainEncoder( ...
      fullfile( imdb.imgDir, imdb.imgName( encSelTrain ) ), ...
      imdb.bdBox( encSelTrain, : ) ) ;
  end
  save( conf.encoderPath, '-struct', 'encoder' ) ;
end

% record time
fprintf( '\n ... Done Train Encoder time: %.2f (s)\n', toc );
