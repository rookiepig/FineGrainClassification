% manually fuse different groups

conf = InitConf();
load( conf.imdbPath );

if( ~exist( 'grpInfo', 'var' ) )
  fprintf( 'load grpInfo\n' );
  load( conf.grpInfoPath );
end
if( ~exist( 'grpModel', 'var' ) )
  fprintf( 'load grpModel\n' );
  load( conf.grpModelPath );
end


test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
nSample = length( imdb.clsLabel );
nClass = max( imdb.clsLabel );

% final fusion scores
scores  = zeros( nSample, nClass);

% average
% selGrp = [ 1 2 3 4 5 6 7 8 ];
% for s = 1 : length( selGrp );
%   g = selGrp( s );
%   scores = scores + grpModel{ g }.bayesProb;
% end
% scores = scores ./ length( selGrp );

% try vote
selGrp = [1 2 3 4 5 6 7 8 ];
for s = 1 : length( selGrp );
  g = selGrp( s );
  sampleIdx = ( 1 : nSample )';
  [ ~, grpPred ] = max( grpModel{ g }.bayesProb, [], 2 );
  scores = scores + full( sparse( sampleIdx, grpPred, 1, nSample, nClass ) );
end

[ manuConf, manuAcc ] = ScoreToConf( scores( test, : ), testLab );
fprintf( 'manu Acc: %.2f %%\n', manuAcc );



