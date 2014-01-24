function [ conf ] = InitConf( )
%% InitConf
%  Desc: init configuration 
%  In: 
%  
%  Out:
%    conf -- (struct) all configuration paramters
%%
PrintTab();fprintf( 'function: %s\n', mfilename );

% config libraries
if ( strcmp( computer(), 'GLNXA64' ) )
  run( '~/vlfeat/toolbox/vl_setup' );
  addpath( '~/libsvm/matlab/' );
  addpath( genpath( '~/minConf/') );
end
%-----------------------------------------------
% Manual paramters
%-----------------------------------------------
% dataset [CUB11, CUB10, STDog]
conf.dataset = 'CUB11';
% PrintTab;fprintf( 'Dataset: %s\n', conf.dataset );
% approach prefix
conf.prefix   = 'prob-clr-fv-512-1x1+3x1';
% 10-fold CV (5-fold is worse than 10-fold)
conf.nFold  = 10; 
conf.MAP_INIT_VAL = -100;
%-----------------------------------------------
% flag paramters
%-----------------------------------------------
conf.isDebug   = true;
% use one-vs-one SVM to get prob output
conf.isOVOSVM = false;
% all cluster use same SVM model
conf.isSameSVM = true;
% use cluster prior to get final test scores
% to strong! needs to be improved
conf.useClusterPrior = false; 
%-----------------------------------------------
% Clustering paramters
%-----------------------------------------------
% cluster type: [spectral, tree]
conf.clusterType = 'spectral';
% group 1 --> no cluster
conf.nGroup = 8; % CUB 8 groups; STDog 7 groups;
% each group's cluster number
conf.nCluster = zeros( conf.nGroup, 1 );
for nc = 1 : conf.nGroup
  conf.nCluster( nc ) = 2 ^ ( nc - 1 );
end
%-----------------------------------------------
% Fusion paramters
%-----------------------------------------------
% map method: [svm,reg,softmax]
% to make svm score comparable
conf.mapType = 'softmax';
switch conf.mapType
  case 'reg'
    conf.regLambda = 1;
    % regression kernel [ rbf, linear ]
    conf.regKerType = 'rbf';
  case 'svm'
    conf.mapSVMOPT = [ '-c 10 -t 2 -q' ];
  % case 'softmax'
end
% fusion method: [average, reg, vote(probability)]
conf.fusionType =  'average';
% map feature normalization method [ 'l2', 'l1' ]
conf.mapNormType ='l2';
%-----------------------------------------------
% SVM paramters
%-----------------------------------------------
% cluster svm option
conf.clusterSVMOPT = [ '-c 10 -t 4 -q' ];
% original svm option
conf.orgSVMOPT = [ '-c 10 -t 4 -q' ];
% group svm option
conf.grpSVMOPT = cell( 1, conf.nGroup );
for g = 1 : conf.nGroup
  conf.grpSVMOPT{ g } = sprintf( '-c %f -t 4 -q\n', 10 * conf.nCluster( g ) );
end
if( conf.isOVOSVM )
  % one-vs-one SVM --> libsvm prob
  conf.clusterSVMOPT = [ conf.clusterSVMOPT, ' -b 1' ];
  conf.orgSVMOPT = [ conf.orgSVMOPT, ' -b 1' ];
  for g = 1 : conf.nGroup
    conf.grpSVMOPT{ g } = [ conf.grpSVMOPT{ g }, ' -b 1' ];
  end
end


%-----------------------------------------------
% Path Paramters
%-----------------------------------------------
conf.outDir       = [ 'data/', conf.dataset ];
conf.cacheDir     = [ 'cache/', conf.dataset ];
conf.kernelDir    = [ '../align/data/', conf.dataset ];
%
conf.imdbPath     = fullfile( conf.outDir, 'imdb.mat' );
% kernel types:
%   seg-fv-clr-300-bdbox-kernel
%   bdbox-fv-clr-300-kernel
%   clr_fv_256_1x1+3x1-kernel
%   seg-clr-fv-256-kernel
%   clr-fv-512-1x1+3x1-kernel

conf.kernelPath   = fullfile( conf.kernelDir, ...
  'clr-fv-512-1x1+3x1-kernel.mat'  );

conf.clsSimPath   = fullfile( conf.outDir, [ conf.prefix '-clsSim.mat' ] );
%
conf.grpInfoPath  = fullfile( conf.outDir, [ conf.prefix '-grpInfo.mat' ] );
conf.grpModelPath = fullfile( conf.outDir, [ conf.prefix '-grpModel.mat' ] );
%
conf.fusionPath   = fullfile( conf.outDir, [ conf.prefix '-fusion.mat' ] );
conf.confPath     = fullfile( conf.outDir, [ conf.prefix, '-conf.mat' ] );

% end function InitConf

