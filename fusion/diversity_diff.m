% calculate diversity for all groups

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
nGroup = conf.nGroup;
scores = zeros( nSample, nClass );


% get 0-1 vec for each group
N = zeros( nSample, nGroup );
for g = 1 : nGroup
  [ ~, pred ] = max( fusion.grpProb{ g }, [], 2 );
  N( :, g ) = ( pred == imdb.clsLabel );
end

% get diff histogram
N_test = N( test, : );
h_cnt = sum( N_test, 2 );
h = hist( h_cnt, nGroup + 1 );

% get variance
p = h ./ length( test );
x = ( 0 : nGroup ) ./ nGroup;

E_x = x * p';
Var_x = ( x - E_x ).^2 * p';

fprintf( 'difficulty (var) is %f\n', Var_x );