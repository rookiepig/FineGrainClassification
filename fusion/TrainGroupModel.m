function [ curModel ] = TrainGroupModel( curGrp )
%% TrainGroupModel
%  Desc: train group model according to cluster information
%  In: 
%    curGrp -- (struct) clustering information for one group
%  Out:
%    curModel -- (struct) group model
%%
fprintf( 'function: %s\n', mfilename );

% init configuration
conf = InitConf( );
% load imdb, kernel
load( conf.imdbPath );
fprintf( '\t loading kernel (maybe slow)\n' );
load( conf.kernelPath );

%% Train current group Fusion model
disp( curGrp );

% Stage1: get SVM features
mapFeat = GetSVMFeat( conf, imdb, kernel, curGrp );
% Stage2: train original SVM
[ mapFeat, orgSVM ] =  TrainOrgSVM( conf, imdb, kernel, ...
  curGrp, mapFeat );
% Stage3: train map model
switch conf.mapType
  case 'reg'
    
  case 'svm'
    [ mapModel, scores ] = TrainMapSVM(  conf, imdb, mapFeat );
  otherwise
    fprintf( '\t Error: unknow map method: %s\n', conf.mapType );
end

% set current model
curModel.mapType = conf.mapType;
curModel.mapFeat = mapFeat;
curModel.orgSVM = orgSVM;
curModel.mapModel = mapModel;
curModel.scores = scores;

disp(curModel);

% end function TrainGroupModel