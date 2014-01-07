%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step2_aggre_group.m
% Desc: aggregate each group info and model to one file
% Author: Zhang Kang
% Date: 2014/01/03
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 'Run: %s\n', mfilename );

% get configuration
conf = InitConf( );

% cache file name
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

% aggregate grpInfo
if( ~exist( conf.grpInfoPath, 'file' ) )
  fprintf( '\t aggregate grpInfo\n' );
  grpInfo = cell( 1, conf.nGroup );
  for g = 1 : conf.nGroup
    if( exist( cacheGrpInfo{ g }, 'file' ) )
      % load caecheGrpInfo
      load( cacheGrpInfo{ g } );
      grpInfo{ g } = curGrp;
    else
      fprintf( 'Error: can not find file %s\n', cacheGrpInfo{ g } );
    end
  end
  % save grpInfo
  fprintf( '\t save grpInfo to %s\n', conf.grpInfoPath );
  save( conf.grpInfoPath, 'grpInfo' );
end

% aagregate grpModel
if( ~exist( conf.grpModelPath, 'file' ) )
  fprintf( '\t aggregate grpModel\n' );
  grpModel = cell( 1, conf.nGroup );
  for g = 1 : conf.nGroup
    if( exist( cacheGrpModel{ g }, 'file' ) )
      % load caecheGrpInfo
      load( cacheGrpModel{ g } );
      grpModel{ g } = curModel;
    else
      fprintf( 'Error: can not find file %s\n', cacheGrpModel{ g } );
    end
  end
  % save grpModel
  fprintf( '\t save grpModel to %s\n', conf.grpModelPath );
  save( conf.grpModelPath, 'grpModel' );
end

% end script step2_aggre_group