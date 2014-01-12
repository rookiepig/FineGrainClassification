function [ curModel ] = TrainGroupModel( curGrp )
%% TrainGroupModel
%  Desc: train group model according to cluster information
%  In: 
%    curGrp -- (struct) clustering information for one group
%  Out:
%    curModel -- (struct) group model
%%
PrintTab();fprintf( 'function: %s\n', mfilename );

% init configuration
conf = InitConf( );
% load imdb, kernel
load( conf.imdbPath );
PrintTab();fprintf( '\t loading kernel (maybe slow)\n' );
load( conf.kernelPath );

%% Train current group model
disp( curGrp );

curModel.isOVOSVM = conf.isOVOSVM;
if( conf.isOVOSVM )
  %% use libsvm probability
  PrintTab();fprintf( '\t one-vs-one class SVM to libsvm prob\n' );
  [ probFeat, ovoSVM ] =  TrainProbSVM( conf, imdb, kernel, curGrp );
  curModel.probFeat = probFeat;
  curModel.ovoSVM  = ovoSVM;
else
  PrintTab();fprintf( '\t one-vs-all class SVM\n' );
  % Stage1: get SVM features
  svmScore = GetSVMFeat( conf, imdb, kernel, curGrp );
  % Stage2: train original SVM
  [ svmScore, ovaSVM ] =  TrainOrgSVM( conf, imdb, kernel, ...
    curGrp, svmScore );
  % Stage3: train map model
  % set current model
  curModel.nFold  = conf.nFold;
  curModel.svmScore = svmScore;
  curModel.ovaSVM = ovaSVM;
  curModel.mapType = conf.mapType;
  PrintTab();fprintf( '\t class SVM map type: %s\n', conf.mapType );
  switch conf.mapType
    case 'reg'
      svmScore = NormMapFeat( conf, imdb, svmScore );
      probFeat = TrainMapReg( conf, imdb, svmScore, imdb.clsLabel );
    case 'svm'
      [ mapSVM, probFeat ] = TrainMapSVM(  conf, imdb, svmScore );
      curModel.mapSVM      = mapSVM;
    case 'softmax'
      % bayes combine cluster and class prob
      probFeat           = TrainMapSoft( conf, imdb, curGrp, svmScore );
      curModel.probFeat  = probFeat;
      curModel.bayesProb = BayesCombine( conf, imdb, curGrp, curModel );
    otherwise
      PrintTab();fprintf( '\t Error: unknow map method: %s\n', conf.mapType );
  end
  % set probFeat
  curModel.probFeat = probFeat;
end % end if isOVOSVM

% show curModel
disp(curModel);

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );
% end function TrainGroupModel