%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: show_wrong_test.m
% Desc: show wrong test samples
% Author: Zhang Kang
% Date: 2014/02/13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load( 'data/CUB11/imdb.mat' );
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );
test  = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );

GRP_NUM = numel( grpInfo );


% show error images
for g = 1 : 1
  curProb  = grpModel{ g }.bayesProb;
  [ ~, predLab ] = max( curProb, [], 2 );
  % get train test true false index
  trainTrue = intersect( train, find( predLab == imdb.clsLabel ) );
  trainFalse = intersect( train, find( predLab ~= imdb.clsLabel ) );
  
  testTrue = intersect( test, find( predLab == imdb.clsLabel ) );
  testFalse = intersect( test, find( predLab ~= imdb.clsLabel ) );
  
  toShow = testFalse;
  SHOW_TIME = 150;
  batchNum = ceil( length( toShow ) / SHOW_TIME );
  for t = 1 : SHOW_TIME
    st = ( t - 1 ) * batchNum + 1;
    ed = min( t * batchNum, length( toShow ) );
    imgNum = ( ed - st ) + 1;
    WID = ceil( sqrt( imgNum ) );
    for ii = 1 : imgNum
      imgIdx = toShow( st + ii - 1 );
      fileName = fullfile( imdb.imgDir, imdb.imgName{ imgIdx } );
      subplot( WID, WID, ii ); imagesc( imread( fileName ) );
    end
    fprintf( 'press any key to cont...\n' );
    pause;
  end
end