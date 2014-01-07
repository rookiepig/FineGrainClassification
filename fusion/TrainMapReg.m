function [ scores ] = TrainMapReg( conf, imdb, mapFeat, sampleLab )
%% TrainMapReg
%  Desc: regression to map SVM scores to final scores
%        **map feature must be normalized**
%  In: 
%    conf, imdb -- basic variables
%    mapFeat -- (nSample * nClass) SVM feature with test SVM feat
%    sampleLab -- (nSample * 1) ground truth label
%  Out:
%    scores  -- (nSample * nClass) final score for map SVM
%%

fprintf( '\t function: %s\n', mfilename );
tic;

% init basic variables
nSample = length( sampleLab );
nClass  = max( sampleLab );
train   = find( imdb.ttSplit == 1 );


% get regression mapping scores
trainFeat = mapFeat( train, : );
allFeat  = mapFeat;
lambda = conf.regLambda;

% prepare train and all sample kernel
fprintf( '\t reg kernel: %s\n', conf.regKerType );
switch conf.regKerType
  case 'rbf'
    dist = EuDist2( trainFeat, trainFeat );
    gamma = mean( mean( dist ) ) * 2;
    trainK = ( exp( - dist ./ gamma ) );
    allK = ( exp( - EuDist2( allFeat, trainFeat ) ./ gamma ) );
  case 'linear'
    trainK = trainFeat * trainFeat';
    allK  = allFeat * trainFeat';
  otherwise
    fprintf( '\t unknow reg kernel: %s\n', conf.regKerType );
end




% get 0-1 class label for training sample
trainLab = sampleLab( train );
trainGt = zeros( size( length( train ), nClass ) );

for m = 1 : length( train )
  trainGt( m, trainLab( m ) ) = 1;
end
scores = allK * ( ( trainK + lambda * eye( size( trainK ) ) ) \ trainGt );

fprintf( '\t function: %s -- time: %.2f (s)\n', mfilename, toc );

% end function TrainMapReg
