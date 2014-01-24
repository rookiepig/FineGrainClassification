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

% get Q matrix
Q = zeros( nGroup, nGroup );
for j = 1 : nGroup
  p_j  = N( test, j );
  for k = 1 : nGroup
    p_k = N( test, k );
    n11 = sum( p_j & p_k );
    n10 = sum( p_j & ~p_k );
    n01 = sum( ~p_j & p_k );
    n00 = sum( ~p_j & ~p_k );
    Q( j, k ) = ( n11 * n00 - n01 * n10 ) / ( n11 * n00 + n01 * n10 );
  end
end

% get Q_av
Q_av = 0;
for g = 1 : nGroup - 1
  Q_av = Q_av + sum( Q( g, g + 1 : nGroup ) );
end
Q_av = Q_av * 2 / ( nGroup * ( nGroup - 1 ) );

fprintf( 'Q_av: %f\n', Q_av );


