% combine different group results

% init basic vars
conf = InitConf();
load( conf.imdbPath );
if( ~exist( 'fusion', 'var' ) )
  load( conf.fusionPath );
end

test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
train = find( imdb.ttSplit == 1 );
trainLab = imdb.clsLabel( train );

nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
scores = zeros( nSample, nClass );

% get training accuracy
% trainAcc = zeros( conf.nGroup, 1 );
% for g = 1 : conf.nGroup
%   [ ~, trainAcc( g ) ] = ScoreToConf( fusion.grpProb{ g }( train, : ), ...
%     trainLab );
%   fprintf( 'Cur grp %d -- train Acc %.2f %%\n', g, trainAcc( g ) );
% end

% % l2 norm --> weight
% trainWgt = ( trainAcc - min( trainAcc ) ) / ...
%   max( max( trainAcc ) - min( trainAcc ), 1e-12 );

% trainWgt = exp( trainWgt );

% average fuse
fprintf( '\n' );
selGrp = [ 3 5 6  7 ];
for t = 1 : length( selGrp )
  g = selGrp( t );
  fprintf( 'Cur grp %d -- test Acc %.2f %%\n', g, fusion.grpAcc( g ) );
  scores = scores + fusion.grpProb{ g };
end

% max fuse
% selGrp = [ 1 2 3 4 5 6 7 8 ];
% for t = 1 : length( selGrp )
%   g = selGrp( t );
%   fprintf( 'Cur grp %d -- test Acc %.2f %%\n', g, fusion.grpAcc( g ) );
%   curProb = fusion.grpProb{ g };
%   maxInd = find( curProb( : ) > scores( : ) );
%   scores( maxInd ) = curProb( maxInd );
% end

[ ~, grpAcc ] = ScoreToConf( scores( test, : ), testLab );
fprintf( 'Fuse grp:\n' );
disp( selGrp );
fprintf( 'Acc: %.2f %%\n', grpAcc );
