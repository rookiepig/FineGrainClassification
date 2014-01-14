%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: cmp_group_global.m
% Desc: compare group SVM and global SVM
% Author: Zhang Kang
% Date: 2013/12/29
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load( 'imdb.mat' );
load( 'model.mat' );
if( ~exist( 'kernelTrain', 'var' ) )
  load( 'kernel.mat' );
end

% comprae class and group classes
cmpCls = 71;
% groupCls = [ 59 62 66 71 72 ];
grpCls = [ 71 72 ];


numClasses = numel( imdb.clsName );
train = ( imdb.ttSplit == 1 );
test = ( imdb.ttSplit == 0 ) ;

% prepare kernel matrix
fprintf( 'Preapre kernel matrix\n' );
% group training and testing index
grpTrain = ismember( imdb.clsLabel( train ), grpCls );
grpTest = ismember( imdb.clsLabel( test ), grpCls );

grpTrainK = kernelTrain( :, 2 : end );
grpTestK = kernelTest( :, 2 : end );
allTrainK = kernelTrain( :, 2 : end );
allTestK = kernelTest( :, 2 : end );

numGrpTrain = sum( grpTrain );
numGrpTest = sum( grpTest );

grpTrainK = grpTrainK( grpTrain, grpTrain );
grpTestK = grpTestK( grpTest , grpTrain );
allTrainK = allTrainK( grpTrain, : );
allTestK = allTestK( grpTest, : );

grpTrainK = [ ( 1 : numGrpTrain )', grpTrainK ];
grpTestK = [ ( 1 : numGrpTest )', grpTestK ];
allTrainK = [ ( 1 : numGrpTrain )', allTrainK ];
allTestK = [ ( 1 : numGrpTest )', allTestK ];


% train group SVM and compare with all SVM
for c = cmpCls : cmpCls
  fprintf( 'Class: %s (%.2f %%)\n', imdb.clsName{ c } );
  % train group SVM
  allTrain = imdb.clsLabel( train );
  yTrain = 2 * ( allTrain( grpTrain ) == c ) - 1 ;
  allTest = imdb.clsLabel( test );
  yTest = 2 * ( allTest( grpTest ) == c ) - 1 ;
  grpModel = libsvmtrain( double( yTrain ), double( grpTrainK ), ...
    '-c 0.001 -t 4' ) ;
  % show mA
  fprintf( '\t Group SVM mA:\n' );
  [grpPredCls, acc, grpScore ] = libsvmpredict( double( yTrain ), ...
    double( grpTrainK ), grpModel );
  fprintf( '\t\t Training mA: %.2f %%\n', acc( 1 ) );
  [grpPredCls, acc, grpScore ] = libsvmpredict( double( yTest ), ...
    double( grpTestK ), grpModel );
  fprintf( '\t\t Testing mA: %.2f %%\n', acc( 1 ) );
  % all SVM
  fprintf( '\t All SVM mA:\n' );
  [allPredCls, acc, allScore ] = libsvmpredict( double( yTrain ), ...
    double( allTrainK ), model{ c } );
  fprintf( '\t\t Training mA: %.2f %%\n', acc( 1 ) );
  [allPredCls, acc, allScore ] = libsvmpredict( double( yTest ), ...
    double( allTestK ), model{ c } );
  fprintf( '\t\t Testing mA: %.2f %%\n', acc( 1 ) );
end
