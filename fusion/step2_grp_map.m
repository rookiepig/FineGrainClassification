%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step2_grp_map.m
% Desc: map group model score
% Author: Zhang Kang
% Date: 2014/01/13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step2_grp_map( grpID )

PrintTab();fprintf( 'Run: %s\n', mfilename );

% get configuration
conf = InitConf( );

if( grpID > conf.nGroup )
  PrintTab();fprintf( 'Error: task index esceed maximum group\n' );
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

% load curGrp
if( exist( cacheGrpInfo{ grpID }, 'file' ) )
  PrintTab();
  fprintf( 'Load cacheGrpInfo from file %s\n', cacheGrpInfo{ grpID } );
  load( cacheGrpInfo{ grpID } );
else
  % error
  PrintTab();
  fprintf('Error: can not find grp info %s\n', cacheGrpInfo{ grpID } );
end


% map svm score
if( conf.isSameSVM )
  % all use group 1 SVM
  load( cacheGrpModel{ 1 } );
else
  load( cacheGrpModel{ grpID } );
end

curModel = TrainGrpMap( curGrp, curModel );
save( cacheGrpModel{ grpID }, 'curModel' );

% end script step1_train_group