%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: TrainGrpModel.m
% Desc: train Group Fusion Model (split for each group)
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function TrainGrpModel( taskID )
fprintf( 'Train group model task: %d\n', taskID );

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
% cache output for each group
cacheModel = cell( 1, GRP_SYS_NUM );
for g = 1 : GRP_SYS_NUM
  tmpName = sprintf( '-tmpModel%03d.mat', g );
  cacheModel{ g } = fullfile( conf.cacheDir, [ conf.prefix, tmpName ] );
end

% split 10 fold CV
splitCVFold;

%% Step3: Train Group Fusion model
if( exist( conf.grpModelPath, 'file' ) )
  % load model from file
  fprintf( 'Load all groups model from file %s\n', conf.grpModelPath );
  load( conf.grpModelPath );
else
  g = taskID;
  fprintf( 'Group: %d (%.2f %%)\n', g, 100 * g / GRP_SYS_NUM );
  disp( grp{ g } );

  if( exist( cacheModel{ g }, 'file' ) )
    % load current goup model from file
    fprintf( '\t Load group model %d from file %s\n', g, cacheModel{ g } );
    load( cacheModel{ g } );
    grpModel{ g } = curModel;
  else
    % train each group fusion model
    cacheStage = cell( 1, 3 );
    for s = 1 : 3
      fprintf( '\t Stage%d: ', s );
      tmpName = sprintf( '-tmpStage%03d_%03d', g, s );
      cacheStage{ s } = fullfile( conf.cacheDir, [ conf.prefix, tmpName ] );
      if( exist( cacheStage{ s }, 'file' ) )
        fprintf( '\t Load group model %d stage %d from file %s\n', g, s, cacheStage{ s } );
        % load current stage --> update curModel
        load( cacheStage{ s } );
      else
        switch s
          case 1
            %% Stage1: get map features
            getSVMFeat;
          case 2
            %% Stage2: train original SVM
            trainOrgSVM;
          case 3
            %% Stage3: train map SVM
            trainMapSVM;
        end % end switch stage
      end % end if stage file
    end % end for stage
    fprintf( '\t Save group model %d to file %s\n', g, cacheModel{ g } );
    save( cacheModel{ g }, 'curModel' );
  end % end else load model

end % end if conf.grpModelPath



fprintf( '\n...Done\n' );