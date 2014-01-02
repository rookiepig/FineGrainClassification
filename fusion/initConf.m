%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: initConf.m
% Desc: init configuration for Group Fusion
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 'Init configuration paramters ... \n' );

% config libraries
if ( strcmp( computer(), 'GLNXA64' ) )
  run( '~/vlfeat/toolbox/vl_setup' );
  addpath( '~/libsvm/matlab/' );
end

global conf;

%-----------------------------------------------
% Manual paramters
%-----------------------------------------------
conf.prefix   = 'oracle'
conf.foldNum  = 10;

%-----------------------------------------------
% flag paramters
%-----------------------------------------------
conf.isDebug   = true;
conf.isSVMProb = false;
% use cluster prior to get final test scores
conf.useClusterPrior = true; 
% use oracle cluster
conf.useOracleCluster = true;

%-----------------------------------------------
% Path Paramters
%-----------------------------------------------
conf.outDir       = 'data/';
conf.cacheDir     = 'cache/';
conf.imdbPath     = fullfile( conf.outDir, [ conf.prefix '-imdb.mat' ] );
conf.grpPath      = fullfile( conf.outDir, [ conf.prefix '-grp.mat' ] );
conf.kernelPath   = fullfile( conf.outDir, [ conf.prefix '-kernel.mat' ] );
conf.clsSimPath   = fullfile( conf.outDir, [ conf.prefix '-clsSim.mat' ] );
conf.grpModelPath = fullfile( conf.outDir, [ conf.prefix '-grpModel.mat' ] );
conf.resultPath   = fullfile( conf.outDir, [ conf.prefix, '-result.mat' ] );


%-----------------------------------------------
% SVM paramters
%-----------------------------------------------
conf.orgSVMOPT = [ '-c 10 -t 4 -q' ];
if( conf.isSVMProb )
  conf.orgSVMOPT = [ conf.orgSVMOPT, ' -b 1' ];
end
conf.mapSVMOPT = [ '-c 10 -t 2 -q' ];

fprintf( '\n...Done\n' );


