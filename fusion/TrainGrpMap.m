function [ curModel ] = TrainGrpMap( curGrp, curModel )
%% TrainGrpMap
%  Desc: map svm score
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

%% Train current group model
disp( curGrp );

if( conf.isOVOSVM )
  %% one-vs-one svm
  PrintTab();fprintf( '\t one-vs-one SVM no need to map\n' );
else
  %% one-vs-all svm mapping
  PrintTab();fprintf( '\t one-vs-all SVM mapping\n' );
  curModel.mapType = conf.mapType;
  PrintTab();fprintf( '\t map type: %s\n', conf.mapType );
  svmScore = curModel.svmScore;
  switch conf.mapType
    case 'reg'
      svmScore = NormMapFeat( conf, imdb, svmScore );
      probFeat = TrainMapReg( conf, imdb, svmScore, imdb.clsLabel );
    case 'svm'
      [ mapSVM, probFeat ] = TrainMapSVM(  conf, imdb, svmScore );
      curModel.mapSVM      = mapSVM;
    case 'softmax'
      % bayes combine cluster and class prob
      [ wSoftmax, probFeat ] = TrainMapSoft( conf, imdb, curGrp, svmScore );
      curModel.bayesProb     = probFeat;
      curModel.wSoftmax      = wSoftmax;
      % old version
      % curModel.probFeat  = probFeat;
      % curModel.bayesProb = BayesCombine( conf, imdb, curGrp, curModel );
    otherwise
      PrintTab();fprintf( '\t Error: unknow map method: %s\n', conf.mapType );
  end
  % set probFeat
  curModel.probFeat = probFeat;
end % end if isOVOSVM

% show curModel
disp(curModel);

PrintTab();fprintf( 'function: %s -- time: %.2f (s)\n', mfilename, toc );
% end function TrainGrpMap