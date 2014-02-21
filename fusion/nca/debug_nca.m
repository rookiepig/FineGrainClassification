% debug nca
addpath( './nca/' );

nSample = 10;
nClass  = 2;

X = rand( nSample, nClass );
Y = full( sparse( 1 : nSample, randi( nClass, [ nSample, 1 ] ), 1 ) );

K = 2; fprintf( 'knn K = %d', K );

idx = knnsearch( X, X, 'K', K );
idx = idx';

A = eye( nClass, nClass );

[ F, dF ] = nca_obj_knn( A( : ), X, Y, idx );


