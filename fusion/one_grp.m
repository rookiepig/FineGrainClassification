%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: one_grp.m
% Desc: tune one grp's results
% Author: Zhang Kang
% Date: 2014/02/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load( 'data/CUB11/imdb.mat' );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );
test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
nCluster = curGrp.nCluster;
nSample  = length( imdb.clsLabel );
nClass   = max( imdb.clsLabel );

% max cluster selection
maxClusterProb = zeros( nSample, nClass ) - 100;
for t = 1 : nSample
  ct     = curGrp.clsToCluster( t );
  grpCls = curGrp.cluster{ ct };
  maxClusterProb( t, grpCls ) = curModel.svmScore( t, grpCls );
end % end for nSample
[ ~, testAcc ] = ScoreToConf( maxClusterProb( test, : ), testLab );
fprintf( 'Max cluster test acc: %.2f %%\n', testAcc );




% % ova SVM  score distribution
% maxTrainScore = max( grpModel{ 1 }.svmScore( train, : ), [], 2 );
% sp = linspace( min( maxTrainScore ), max( maxTrainScore ), 100 );
% hTrain = histc( maxTrainScore, sp );
% figure; bar( sp, hTrain ); title( 'Train Score Distr' );
% 
% maxTestScore = max( grpModel{ 1 }.svmScore( test, : ), [], 2 );
% sp = linspace( min( maxTestScore ), max( maxTestScore ), 100 );
% hTest = histc( maxTestScore, sp );
% figure; bar( sp, hTest ); title( 'Test Score Distr' );