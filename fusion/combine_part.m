% load each part result
if( ~exist( 'part', 'var' ) )
  part = cell( 1, 4 );
  for ii = 1 : 4
    fName = sprintf( 'data/prob-org-part%d-fusion.mat', ii );
    fprintf( 'load %s\n', fName );
    load( fName );
    part{ ii } = fusion;
  end
end

% init basic vars
conf = InitConf();
load( conf.imdbPath );
test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );
nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
scores = zeros( nSample, nClass );

% fuse parts
selPart = [ 1  2 3 4 ];
for t = 1 : length( selPart )
  p = selPart( t );
  fprintf( 'Cur part %d -- Acc %.2f %%\n', p, part{ p }.meanAcc );
  scores = scores + part{ p }.scores;
end

[ ~, partAcc ] = ScoreToConf( scores( test, : ), testLab );
fprintf( 'Fuse part:\n' );
disp( selPart );
fprintf( 'Acc: %.2f %%\n', partAcc );
