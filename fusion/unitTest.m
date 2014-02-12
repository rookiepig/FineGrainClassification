%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: unitTest.m
% Desc: unit test SoftmaxLoss2_all
% Author: Zhang Kang
% Date: 2014/02/05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toyCase = 2;

switch( toyCase )
  case 1
    SAMPLE_NUM  = 10000;
    CLS_NUM     = 200;
    CLUSTER_NUM = 10;

    X = cell( 1, CLUSTER_NUM );
    for t = 1 : CLUSTER_NUM
      X{ t } = rand( SAMPLE_NUM, CLS_NUM );
      k{ t } = CLS_NUM;
    end
    prior = rand( SAMPLE_NUM, CLUSTER_NUM );
    y = ones( SAMPLE_NUM, 1 );
    c2c = ones( CLS_NUM, CLUSTER_NUM );

    wAll = zeros( CLUSTER_NUM * CLS_NUM * ( CLS_NUM - 1 ), 1 );

    profile on;
    [ J, g ] = SoftmaxLoss2_all( wAll, X, y, k, prior, c2c );
    profile viewer;
    
  case 2
    tmp = eye( 4 );
    X{ 1 } = [ ones( 4, 1 ), tmp( :, 1 : 3 ) ];
    X{ 2 } = [ ones( 4, 1 ), tmp( :, 2 : 4 ) ];  
    k{ 1 } = 3; k{ 2 } = 3;
    prior = [ 1 0; 0.8 0.2; 0.2 0.8; 0 1 ];
    y = ( 1 : 4 )';
    c2c = [ 1 0; 1 1; 1 1; 0 1 ];

    wAll = zeros( 16, 1 );
    [ J, g ] = SoftmaxLoss2_all( wAll, X, y, k, prior, c2c );
end


