%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step1_train_group.m
% Desc: train group model (can be splitted to each group)
% Author: Zhang Kang
% Date: 2014/01/03
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step1_train_group( grpID )

PrintTab();fprintf( 'Run: %s\n', mfilename );

% get configuration
conf = InitConf( );

if( grpID > conf.nGroup )
  PrintTab();fprintf( 'Error: task index esceed maimum group\n' );
  return;
end

cacheGrpInfo = cell( 1, conf.nGroup );
for g = 1 : conf.nGroup
  fName = sprintf( '-tmpGrpInfo%03d.mat',  g );
  cacheGrpInfo{ g } = fullfile( conf.cacheDir, [ conf.prefix, fName ] );
end
cacheGrpModel = cell( 1, conf.nGroup );
for g = 1 : conf.nGroup
  fName = sprintf( '-tmpGrpModel%03d.mat',  g );
  cacheGrpModel{ g } = fullfile( conf.cacheDir, [ conf.prefix, fName ] );
end

% curGrp
if( exist( conf.grpInfoPath, 'file' ) )
  % get curGrp from grpInfo
  PrintTab();fprintf( 'Load grpInfo from file %s\n', conf.grpInfoPath );
  load( conf.grpInfoPath );
  curGrp = grpInfo{ grpID };
else
  if( exist( cacheGrpInfo{ grpID }, 'file' ) )
    % load curGrp
    PrintTab();
    fprintf( 'Load cacheGrpInfo from file %s\n', cacheGrpInfo{ grpID } );
    load( cacheGrpInfo{ grpID } );
  else
    % get curGrp by clusteering
    nCluster = conf.nCluster( grpID );
    % clustering
    curGrp   = GroupClustering( nCluster );
    % cluster labeling
    curGrp  = TrainClusterModel( curGrp );
    save( cacheGrpInfo{ grpID }, 'curGrp' );
  end
end

% curModel
if( ~exist( conf.grpModelPath, 'file' ) && ...
  ~exist( cacheGrpModel{ grpID }, 'file' ) )
  % get curModel
  curModel = TrainGroupModel( curGrp );
  save( cacheGrpModel{ grpID }, 'curModel' );
end


% end script step1_train_group