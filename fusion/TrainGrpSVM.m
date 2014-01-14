function [ curModel ] = TrainGrpSVM( curGrp )
%% TrainGrpSVM
%  Desc: train group SVM according to cluster information
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
  %% start
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
end % end if( conf.isOVOSVM )

% show curModel
disp(curModel);

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );
% end function TrainGrpSVM