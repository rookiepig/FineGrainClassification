%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: initConf
% Desc: initial configuration paramters for classification
% Author: Zhang Kang
% Date: 2013/12/05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 1, '\n Init Configuration Paramters ... \n' );

if ( strcmp( computer(), 'GLNXA64' ) )
  % run vl_setup explicitly on Linux platform
  run( '~/vlfeat/toolbox/vl_setup' );
  addpath( '~/libsvm/matlab/' );
  addpath( 'cq/' );
end

% declare global variable
%   conf - configuration paramters
%   imdb - image database

global conf imdb;

% lite version for debug
conf.lite = false;
% dataset name
conf.dataset = 'CUB11';
% name prefix for all output
conf.prefix  = 'seg-cm-clr-300-bdbox';             
if( conf.lite )
  conf.prefix = [ conf.prefix '-lite' ];
end
% enable left-right flip
conf.isLRFlip = true;
% standarize max pix < 1e5            
conf.isStandImg = true;                  

% !!! conflict with seg mask !!
% to handle seg mask
% use nearest neigbour inter
if( conf.isStandImg )
  % conf.maxPixNum = 1e5;
  conf.maxImgSz = 300;
end
% remove zero value feature --> avoid NaN in FV
conf.removeZeroFeat = true;
% enable crop of bounding box
conf.useBoundingBox = true;
% enable segment mask      
conf.useSegMask = true;
% use color moment
conf.useColorMoment = true;
if( conf.useColorMoment )
  % get cm paramters
  load_param;
  conf.cmParam = param;
end

%-----------------------------------------------
% feature&encoder paramters
%-----------------------------------------------
conf.encoderParam = { 'type', 'fv', ...
  'numWords', 256, ...
  'layouts', {'1x1'}, ...                 % spatial pyramid layouts
  'numPcaDimensions', 64, ...              % PCA dimenssion PCA FLAG
  'whitening', false, ...                  % PCA whiten PCA FLAG
  'whiteningRegul', 0.01, ...              % PCA whiten + regularize
  'renormalize', true, ...                 % PCA l2 renormalize
  'seed', 1
  };                                       % encoder paramter
conf.featDimPerChannel = 128;              % if use multiple channel
if( conf.useColorMoment )
  conf.featDimPerChannel = 96;              % if use multiple channel
end
conf.featParam = { 'Sizes' [ 4 6 8 10 ], ...
  'Step', 3, ...
  'Color', 'opponent', ...
  'FloatDescriptors', true };
% PHOW paramter
if( conf.useSegMask )
  conf.maskType = [ 64 / 255, 128 / 255, 192 / 255, 255 / 255 ];
end
%-----------------------------------------------
% model paramters
%-----------------------------------------------
conf.svm.C = 10;
conf.svm.kernel = 'linear';


conf.randSeed = 1 ;                       % initial random seed
randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

%-----------------------------------------------
% path paramters
%-----------------------------------------------
conf.outDir  = [ 'data/'  conf.dataset ];                    % output direcotry and files
conf.imdbPath = fullfile(conf.outDir, [ conf.prefix '-imdb.mat']);
conf.encoderPath = fullfile(conf.outDir, [conf.prefix '-encoder.mat']);
conf.modelPath = fullfile(conf.outDir, [conf.prefix '-model.mat']);
conf.resultPath = fullfile(conf.outDir, [conf.prefix '-result.mat']);
% precompute kernel
conf.kernelPath = fullfile(conf.outDir, [conf.prefix '-kernel.mat']);


fprintf( 1, '\n ... Done\n' );


