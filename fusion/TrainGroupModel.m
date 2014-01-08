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

%% Train current group model
disp( curGrp );

curModel.isSVMProb = conf.isSVMProb;
if( conf.isSVMProb )
  %% use libsvm probability
  fprintf( '\t use libsvm probability\n' );
  [ probFeat, probSVM ] =  TrainProbSVM( conf, imdb, kernel, curGrp );
  curModel.probFeat = probFeat;
  curModel.probSVM  = probSVM;
else
  % Stage1: get SVM features
  mapFeat = GetSVMFeat( conf, imdb, kernel, curGrp );
  % Stage2: train original SVM
  [ mapFeat, orgSVM ] =  TrainOrgSVM( conf, imdb, kernel, ...
    curGrp, mapFeat );
  % Stage3: train map model
  % set current model
  curModel.mapType = conf.mapType;
  curModel.nFold  = conf.nFold;
  curModel.mapFeat = mapFeat;
  curModel.orgSVM = orgSVM;

  switch conf.mapType
    case 'reg'
      mapFeat    = NormMapFeat( conf, imdb, mapFeat );
      [ scores ] = TrainMapReg( conf, imdb, mapFeat, imdb.clsLabel );
      curModel.scores = scores;
    case 'svm'
      [ mapSVM, scores ] = TrainMapSVM(  conf, imdb, mapFeat );
      curModel.mapSVM = mapSVM;
      curModel.scores = scores;
    otherwise
      fprintf( '\t Error: unknow map method: %s\n', conf.mapType );
  end

end % end if isSVMProb

% show curModel
disp(curModel);

fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );
% end function TrainGroupModel