%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: aggreResults.m
% Desc: aggregate all group model and generate final results
% Author: Zhang Kang
% Date: 2014/01/02
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 'Aggregate all cache group model and generate final results\n' );

%% Step1: init configuration
initConf;
disp( conf );

%% Step2: load data
loadData;

%-----------------------------------------------
% Init basic variables
%-----------------------------------------------
numClasses = numel( imdb.clsName );
numSample  = numel( imdb.clsLabel );
train = find( imdb.ttSplit == 1 );
test = find( imdb.ttSplit == 0 ) ;
GRP_SYS_NUM = numel( grp );
grpModel = cell( 1, GRP_SYS_NUM );
% cache output for each group
cacheModel = cell( 1, GRP_SYS_NUM );
for g = 1 : GRP_SYS_NUM
  tmpName = sprintf( '-tmpModel%03d.mat', g );
  cacheModel{ g } = fullfile( conf.cacheDir, [ conf.prefix, tmpName ] );
end

%% Step3: Train Group Fusion model
if( exist( conf.grpModelPath, 'file' ) )
  % load model from file
  fprintf( 'Load all groups model from file %s\n', conf.grpModelPath );
  load( conf.grpModelPath );
else
  % load all cached model
  for g = 1 : GRP_SYS_NUM
    if( exist( cacheModel{ g }, 'file' ) )
      % load current goup model from file
      fprintf( '\t Load group model %d from file %s\n', g, cacheModel{ g } );
      load( cacheModel{ g } );
      grpModel{ g } = curModel;
    else
      fprintf( '\t Error: miss group model %d file %s\n', g, cacheModel{ g } );
    end
  end
  % save all groups' model
  fprintf( 'Save group model to file %s\n', conf.grpModelPath );
  save( conf.grpModelPath, 'grpModel' );
end

% get confusion output and save results
genConfusion;


fprintf( '\n...Done\n' );


