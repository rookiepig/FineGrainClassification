%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: calc_entropy.m
% Desc: calculate each sample's entropy
% Author: Zhang Kang
% Date: 2014/02/12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
conf = InitConf( );
load( 'data/CUB11/imdb.mat' );
if( ~exist( 'grpInfo', 'var' ) )
  load( conf.grpInfoPath );
end
if( ~exist( 'grpModel', 'var' ) )
  load( conf.grpModelPath );
end

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
  % prob = grpInfo{ g }.clusterProb;
  prob = grpModel{ g }.bayesProb;
  entro( :, g ) = - sum( bsxfun( @times, prob, log( prob ) ), 2 );
end 


% draw entropy distribution
for g = 1 : GRP_NUM
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
  
  % save output5
  %saveas( fig, sprintf( 'img/grp-%03d.png', g ) );
  fprintf( 'Press any key to continue...\n' );
  pause;
  close( fig );
end

% 
% % get entropy weighted 
% for t = 1 : 11
%   if( t == 1 )
%     % add 1 cluster probability 
%     fprintf( 'Group 001\n' );
%     finalProb =  grpModel{ 1 }.bayesProb;
%     % oneFuseProb = grpModel{ 1 }.bayesProb;
%   else
%     % each 5 subgroup
%     gSt = 5 * t - 8;
%     gEd = 5 * t - 4;
%     fprintf( 'Group %03d-%03d -- Cluster Num %d\n', gSt, gEd, grpInfo{ gSt }.nCluster );
%     % oneFuseProb = oneFuseProb + grpModel{ gSt }.bayesProb;
%     wgtProb   = zeros( nSample, nClass );
%     noWgtProb = zeros( nSample, nClass );
%     wgt = zeros( nSample, 1 );
%     for g = gSt : gEd
%       maxEntro = log( grpInfo{ g }.nCluster );
%       % wgt = ( maxEntro - entro( :, g ) ) ./ max( maxEntro, eps  );
%       wgt = exp( - entro( :, g ) );
%       wgtProb = wgtProb + bsxfun( @times, wgt, grpModel{ g }.bayesProb );
%       noWgtProb = noWgtProb + grpModel{ g }.bayesProb;
%     end
% 
%     % normalize weight prob
%     wgtProb = bsxfun( @rdivide, wgtProb, sum( wgtProb, 2 ) );
%     finalProb = finalProb + wgtProb;
% 
%     % each subgroup acc
%     [ ~, noWgtAcc( t )  ] = ScoreToConf( noWgtProb( test, : ), testLab );
%     fprintf( '  No Wgt Acc: %.2f %%\n', noWgtAcc( t ) );
% 
%     [ ~, wgtAcc( t ) ] = ScoreToConf( wgtProb( test, : ), testLab );
%     fprintf( '  Entro Wgt Acc: %.2f %%\n', wgtAcc( t ) );
%   end
%   [ ~, finalAcc( t ) ] = ScoreToConf( finalProb( test, : ), testLab );
%   fprintf( 'Combined Acc: %.2f %%\n\n', finalAcc( t ) );
% %   [ ~, oneFuseAcc( t ) ] = ScoreToConf( oneFuseProb( test, : ), testLab );
% %   fprintf( 'Fuse One Acc: %.2f %%\n\n', oneFuseAcc( t ) );
% end
% fprintf( 'Mean Fusion Acc: %.2f %%\n', fusion.meanAcc );
% 

% get each class mean entropy
% clsMeanEntro = zeros( nClass, 1 );
% for c = 1 : nClass
%   curIdx = intersect( find( imdb.clsLabel == c ), train );
%   clsMeanEntro( c ) = mean( curEntro( curIdx ) );
% end


% % entro threshold score
% ENTRO_THRES = 1;
% 
% thresScore = zeros( nSample, nClass );
% relia = find( entro( :, 1 ) <= ENTRO_THRES );
% noRelia = find( entro( :, 1 ) > ENTRO_THRES );
% thresScore( relia, : )   = grpModel{ 1 }.svmScore( relia, : );
% thresScore( noRelia, : ) = fusion.scores( noRelia, : );
% 
% [ ~, trainAcc ] = ScoreToConf( thresScore( train, : ), trainLab );
% fprintf( 'Thres train acc: %.2f %%\n', trainAcc );
% [ ~, testAcc ] = ScoreToConf( thresScore( test, : ), testLab );
% fprintf( 'Thres test acc: %.2f %%\n', testAcc );




