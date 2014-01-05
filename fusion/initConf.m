function [ conf ] = InitConf( )
%% InitConf
%  Desc: init configuration 
%  In: 
%  
%  Out:
%    conf -- (struct) all configuration paramters
%%
fprintf( 'function: %s\n', mfilename );
% config libraries
if ( strcmp( computer(), 'GLNXA64' ) )
  run( '~/vlfeat/toolbox/vl_setup' );
  addpath( '~/libsvm/matlab/' );
end
%-----------------------------------------------
% Manual paramters
%-----------------------------------------------
conf.prefix   = 'nocluster';
conf.nFold  = 10;
conf.MAP_INIT_VAL = -100;
%-----------------------------------------------
% flag paramters
%-----------------------------------------------
conf.isDebug   = true;
conf.isSVMProb = false;
% use cluster prior to get final test scores
conf.useClusterPrior = false; 
%-----------------------------------------------
% Clustering paramters
%-----------------------------------------------
% cluster type: [spectral, tree]
conf.clusterType = 'spectral';
% group 1 --> no cluster
conf.nGroup = 6;
%-----------------------------------------------
% Fusion paramters
%-----------------------------------------------
% map method: [svm,reg]
conf.mapType = 'reg';
switch conf.mapType
  case 'reg'
    conf.regLambda = 1;
  case 'svm'
    conf.mapSVMOPT = [ '-c 10 -t 2 -q' ];
end
% fusion method: [average]
conf.fusionType =  'average';
% map feature normalization method [ 'l2', 'l1' ]
conf.mapNormType ='l2';
%-----------------------------------------------
% SVM paramters
%-----------------------------------------------
conf.clusterSVMOPT = [ '-c 10 -t 4 -q' ];
conf.orgSVMOPT = [ '-c 10 -t 4 -q' ];
if( conf.isSVMProb )
  conf.orgSVMOPT = [ conf.orgSVMOPT, ' -b 1' ];
end


%-----------------------------------------------
% Path Paramters
%-----------------------------------------------
conf.outDir       = 'data/';
conf.cacheDir     = 'cache/';
%
conf.imdbPath     = fullfile( conf.outDir, 'imdb.mat' );
conf.kernelPath   = fullfile( conf.outDir, 'kernel.mat'  );
conf.clsSimPath   = fullfile( conf.outDir, [ conf.prefix '-clsSim.mat' ] );
%
conf.grpInfoPath  = fullfile( conf.outDir, [ conf.prefix '-grpInfo.mat' ] );
conf.grpModelPath = fullfile( conf.outDir, [ conf.prefix '-grpModel.mat' ] );
%
conf.fusionPath   = fullfile( conf.outDir, [ conf.prefix '-fusion.mat' ] );
conf.resultPath   = fullfile( conf.outDir, [ conf.prefix, '-result.mat' ] );

% end function InitConf

