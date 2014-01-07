function fusion = GroupFusion( conf, imdb, grpInfo, grpModel )
%% GroupFusion
%  Desc: fuse all group's results
%  In: 
%    conf, imdb -- basic variables
%    grpInfo  -- (struct) clustering info of all groups
%    grpModel -- (struct) all group models
%  Out:
%    fusion -- (struct) fusion model and results
%%

fprintf( 'function: %s\n', mfilename );

test = find( imdb.ttSplit == 0 );

if( conf.useClusterPrior )
  % set non-cluster scores to negative infinite
  testNum = length( test );
  nClass = max( imdb.clsLabel );
  for g = 1 : conf.nGroup
    cTc = grpInfo{ g }.clsToCluster( test );
    % set other cluster score to minimum
    for t = 1 : testNum
      clsIdx = grpInfo{ g }.cluster{ cTc( t ) };
      clsIdx = setdiff( ( 1 : nClass )', clsIdx );
      grpModel{ g }.scores( t, clsIdx ) = -1e10;
    end
  end
end

%% Fusion Group Model
fprintf( '\t fusion type: %s\n', conf.fusionType );
switch conf.fusionType
  case 'average'
    % average SVM scores
    scores = zeros( size( grpModel{ 1 }.scores ) );
    for g = 1 : conf.nGroup
      scores = scores + grpModel{ g }.scores;
    end
    scores = scores ./ conf.nGroup;
  case 'reg'
    % concatante all mapFeat and re mapping using regression
    allFeat = [];
    for g = 1 : conf.nGroup
      fprintf( '\t group %d\n', g );
      tmpFeat = grpModel{ g }.mapFeat;
      tmpFeat   = NormMapFeat( conf, imdb, tmpFeat );
      tmpScores = TrainMapReg( conf, imdb, tmpFeat, imdb.clsLabel );
      [ ~, tmpAcc ] = ScoreToConf( tmpScores( test, : ), imdb.clsLabel( test ) );
      fprintf( '\t tmp acc: %.2f %%\n', tmpAcc );
      % concatenate all SVM scores
      allFeat = [ allFeat tmpFeat ];
    end
    scores = TrainMapReg( conf, imdb, allFeat, imdb.clsLabel );
  otherwise
    fprintf( '\t Error: unknown fusion type: %s\n', conf.fusionPath );
    return;
end

[ confusion, meanAcc ] = ScoreToConf( scores( test, : ), imdb.clsLabel( test ) );

% set fusion struct
fusion.scores    = scores;
fusion.confusion = confusion;
fusion.meanAcc   = meanAcc;

fprintf( '\t fusion acc: %.2f %%\n', meanAcc );

% end function GroupFusion


