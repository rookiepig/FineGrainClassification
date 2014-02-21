% test nca
addpath( './nca/' );
addpath( genpath( '~/minConf/') );

load( 'tmp/CUB11-tmpGrpModel001.mat' );
load( 'data/CUB11/imdb.mat' );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );
test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );

trainScore = curModel.svmScore( train, : );
allScore   = curModel.svmScore;

[ Anew, probFeat ] = NCAProb_knn( trainScore, trainLab, allScore, 1000 );

[ ~, trainAcc ] = ScoreToConf( probFeat( train, : ), trainLab );
fprintf( 'nca train acc: %.2f %%\n', trainAcc );

[ ~, testAcc ] = ScoreToConf( probFeat( test, : ), testLab );
fprintf( 'nca test acc: %.2f %%\n', testAcc );