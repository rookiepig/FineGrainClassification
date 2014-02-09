%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: unitTest.m
% Desc: unit test SoftmaxLoss2_all
% Author: Zhang Kang
% Date: 2014/02/05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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