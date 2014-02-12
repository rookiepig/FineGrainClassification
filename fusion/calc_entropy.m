%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: calc_entropy.m
% Desc: calculate each sample's entropy
% Author: Zhang Kang
% Date: 2014/02/12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load( 'data/CUB11/imdb.mat' );
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );
test  = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );

GRP_NUM = numel( grpInfo );

% get sample entropy
entro = zeros( nSample, GRP_NUM );
for g = 1 : GRP_NUM
  prob = grpInfo{ g }.clusterProb;
  entro( :, g ) = - sum( bsxfun( @times, prob, log( prob ) ), 2 );
end 


% draw entropy distribution
for g = 2 : GRP_NUM
  curEntro = entro( :, g );
  minE = min( curEntro );
  maxE = max( curEntro );
  entroSp = linspace( minE, maxE, 100 );
  curProb  = grpModel{ g }.bayesProb;
  [ ~, predLab ] = max( curProb, [], 2 );
  % get train test true false index
  trainTrue = intersect( train, find( predLab == imdb.clsLabel ) );
  trainFalse = intersect( train, find( predLab ~= imdb.clsLabel ) );
  
  testTrue = intersect( test, find( predLab == imdb.clsLabel ) );
  testFalse = intersect( test, find( predLab ~= imdb.clsLabel ) );
  
  % draw hist
  fig = figure;
  h1 = histc( curEntro( trainTrue ), entroSp );
  h2 = histc( curEntro( trainFalse ), entroSp );
  subplot( 1, 2, 1 ); bar( entroSp, [ h1 h2 ] ); 
  title( sprintf( 'Group %d (%d cluster) -- Train', g, grpInfo{ g }.nCluster ) ); 
  legend( 'True', 'False' );
  h3 = histc( curEntro( testTrue ), entroSp );
  h4 = histc( curEntro( testFalse ), entroSp );
  subplot( 1, 2, 2 ); bar( entroSp, [ h3 h4 ] ); 
  title( sprintf( 'Group %d (%d cluster ) -- Test', g, grpInfo{ g }.nCluster ) );
  legend( 'True', 'False' );
  
  % save output
  saveas( fig, sprintf( 'img/grp-%03d.png', g ) );
%   fprintf( 'Press any key to continue...\n' );
%   pause;
  close( fig );
end



% % get entropy weighted 
% entro( :, 1 ) = min( entro( :, 2 : end ), [], 2 );
% 
% entroProb = zeros( nSample, nClass );
% for g = 1 : GRP_NUM
%   entroProb = entroProb + bsxfun( @times, 1 ./ entro( :, g ), grpModel{ g }.bayesProb );
% end
% 
% % train test acc
% [ ~, trainAcc ] = ScoreToConf( entroProb( train, : ), trainLab );
% fprintf( 'train Acc: %.2f %%\n', trainAcc );
% [ ~, testAcc ] = ScoreToConf( entroProb( test, : ), testLab );
% fprintf( 'test Acc: %.2f %%\n', testAcc );

