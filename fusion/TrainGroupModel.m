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
% set current model
curModel.mapType = conf.mapType;
curModel.mapFeat = mapFeat;
curModel.orgSVM = orgSVM;

switch conf.mapType
  case 'reg'
    [ scores ] = TrainMapReg( conf, imdb, mapFeat );
    curModel.scores = scores;
  case 'svm'
    [ mapSVM, scores ] = TrainMapSVM(  conf, imdb, mapFeat );
    curModel.mapSVM = mapSVM;
    curModel.scores = scores;
  otherwise
    fprintf( '\t Error: unknow map method: %s\n', conf.mapType );
end



disp(curModel);

% end function TrainGroupModel