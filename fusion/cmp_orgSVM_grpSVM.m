% compare original SVM vs group SVM
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
nClass = max( imdb.clsLabel );


% compare all group 
%
% for g = 1 : 8
%   fprintf( 'group %d\n', g );
%   cmpAcc = zeros( grpInfo{ g }.nCluster, 2 );
%   for c = 1 : grpInfo{ g }.nCluster
%     grpCls = grpInfo{ g }.cluster{ c };
%     PrintTab();fprintf( 'Cluster %d\n', c );
%     % compare each cluster accuracy
%     tmpIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
%       test );
%     otherIdx = setdiff( ( 1 : nClass ), grpCls );
%     tmpGtLab = imdb.clsLabel( tmpIdx );
%     
%     allScores = grpModel{ 1 }.svmScore( tmpIdx, : );
%     curScores = grpModel{ g }.svmScore( tmpIdx, : );
%     
%     allScores( :, otherIdx ) = -1e10;
%     curScores( :, otherIdx ) = -1e10;
%     
%     [ ~, allLab ] = max( allScores, [], 2 );
%     [ ~, curLab ] = max( curScores, [], 2 );
%     
%     cmpAcc( c, 1 ) = 100 * sum( allLab == tmpGtLab ) / length( tmpIdx );
%     cmpAcc( c, 2 ) = 100 * sum( curLab == tmpGtLab ) / length( tmpIdx );
%     PrintTab();fprintf( 'org acc: %.2f %% -- grp acc: %.2f %%\n', ...
%       cmpAcc( c, 1 ), cmpAcc( c, 2 ) );
%   end
%   oldAcc( g ) = mean( cmpAcc( :, 1 ) );
%   grpAcc( g ) = mean( cmpAcc( :, 2 ) );
%   PrintTab();fprintf( 'mean org: %.2f %% -- mean group %d: %.2f %%\n', ...
%     mean( cmpAcc( :, 1 ) ), g, mean( cmpAcc( :, 2 ) ) );
% end


% compare one group

cmpAcc = zeros( curGrp.nCluster, 2 );
for c = 1 : 1
  grpCls = [ 1 2 3 4 5 ];
  PrintTab();fprintf( 'Cluster %d\n', c );
  % compare each cluster accuracy
  tmpIdx = intersect( find( ismember( imdb.clsLabel, grpCls ) ), ...
    test );
  otherIdx = setdiff( ( 1 : nClass ), grpCls );
  tmpGtLab = imdb.clsLabel( tmpIdx );

  allScores = grpModel{ 1 }.svmScore( tmpIdx, : );
  curScores = curModel.svmScore( tmpIdx, : );

  allScores( :, otherIdx ) = -1e10;
  curScores( :, otherIdx ) = -1e10;

  [ ~, allLab ] = max( allScores, [], 2 );
  [ ~, curLab ] = max( curScores, [], 2 );

  cmpAcc( c, 1 ) = 100 * sum( allLab == tmpGtLab ) / length( tmpIdx );
  cmpAcc( c, 2 ) = 100 * sum( curLab == tmpGtLab ) / length( tmpIdx );
  PrintTab();fprintf( 'org acc: %.2f %% -- grp acc: %.2f %%\n', ...
    cmpAcc( c, 1 ), cmpAcc( c, 2 ) );
end
% oldAcc = mean( cmpAcc( :, 1 ) );
% grpAcc = mean( cmpAcc( :, 2 ) );
% PrintTab();fprintf( 'mean org: %.2f %% -- mean group %d: %.2f %%\n', ...
%   mean( cmpAcc( :, 1 ) ), g, mean( cmpAcc( :, 2 ) ) );

