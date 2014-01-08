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

nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
test = find( imdb.ttSplit == 0 );

grpConf   = cell( 1, conf.nGroup );
grpAcc    = zeros( 1, conf.nGroup );
scores = zeros( nSample, nClass);

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
fusion.isSVMProb = conf.isSVMProb;
if( conf.isSVMProb )
  %% probability fusion
  % bayes output for each group
  fprintf( '\t get bayes prob for each group\n' );
  grpProb   = cell( 1, conf.nGroup );

  for g = 1 : conf.nGroup
    fprintf( '\t Group %d\n', g );
    % init current group final probability
    grpProb{ g } = zeros( nSample, nClass );
    for t = 1 : grpInfo{ g }.nCluster
      grpCls = grpInfo{ g }.cluster{ t };
      % get cluster prior prob
      clusterProb = grpInfo{ g }.clusterProb( :, t );
      for c = 1 : length( grpCls )
        curCls = grpCls( c );
        % get class likelihood prob
        clsProb = grpModel{ g }.probFeat( :, curCls );
        % bayes rule to get final class prob
        grpProb{ g }( :, curCls ) = grpProb{ g }( :, curCls ) + ...
          clsProb .* clusterProb;
      end % end for each class
    end % end for each cluster
    % get current group confusion and accuracy
    [ grpConf{ g }, grpAcc( g ) ] = ScoreToConf( grpProb{ g }( test, : ), ...
      imdb.clsLabel( test ) );
    fprintf( '\t current acc: %.2f %%\n', grpAcc( g ) );
  end % end for each group

  fusion.grpProb = grpProb;
  fusion.grpConf = grpConf;
  fusion.grpAcc  = grpAcc;

  % fuse each group's probability
  fprintf( '\t fusion type: %s\n', conf.fusionType );
  fusion.fusionType = conf.fusionType;
  switch conf.fusionType
    case 'average'
      for g = 1 : conf.nGroup
        scores = scores + grpProb{ g };
      end
      scores = scores ./ conf.nGroup;
    case 'vote'
      for g = 1 : conf.nGroup
        sampleIdx = ( 1 : nSample )';
        [ ~, grpPred ] = max( grpProb{ g }, [], 2 );
        scores( sampleIdx, grpPred ) = scores( sampleIdx, grpPred ) + 1;
      end
    otherwise
      fprintf( '\t Error: unknown fusion type: %s\n', conf.fusionType );
  end

else

  %% no probability fusion
  fprintf( '\t fusion type: %s\n', conf.fusionType );
  fusion.fusionType = conf.fusionType;
  switch conf.fusionType
    case 'average'
      % average SVM scores
      for g = 1 : conf.nGroup
        scores = scores + grpModel{ g }.scores;
      end
      scores = scores ./ conf.nGroup;
    case 'reg'
      % concatante all mapFeat and re mapping using regression
      allFeat = [];
      grpScore = cell( 1, conf.nGroup );
      for g = 1 : conf.nGroup
        fprintf( '\t group %d\n', g );
        tmpFeat = grpModel{ g }.mapFeat;
        tmpFeat   = NormMapFeat( conf, imdb, tmpFeat );
        grpScore{ g } = TrainMapReg( conf, imdb, tmpFeat, imdb.clsLabel );
        [ grpConf{ g }, grpAcc( g ) ] = ScoreToConf( grpScore{ g }( test, : ), ...
          imdb.clsLabel( test ) );
        fprintf( '\t current acc: %.2f %%\n', grpAcc{ g } );
        % concatenate all SVM scores
        allFeat = [ allFeat tmpFeat ];
      end
      % save each group's result
      fusion.grpScore = grpScore;
      fusion.grpConf = grpConf;
      fusion.grpAcc  = grpAcc;
      % get final mapping
      scores = TrainMapReg( conf, imdb, allFeat, imdb.clsLabel );
    case 'vote'
      % get each group score
      grpScore = cell( 1, nGroup );
      for g = 1 : conf.nGroup
        fprintf( '\t group %d\n', g );
        tmpFeat = grpModel{ g }.mapFeat;
        tmpFeat   = NormMapFeat( conf, imdb, tmpFeat );
        grpScore{ g } = TrainMapReg( conf, imdb, tmpFeat, imdb.clsLabel );
        [ grpConf{ g }, grpAcc( g ) ] = ScoreToConf( grpScore{ g }( test, : ), ...
          imdb.clsLabel( test ) );
        fprintf( '\t current acc: %.2f %%\n', grpAcc{ g } );
      end
      % save each group's result
      fusion.grpScore = grpScore;
      fusion.grpConf = grpConf;
      fusion.grpAcc  = grpAcc;
      % voting
      for g = 1 : conf.nGroup
        sampleIdx = ( 1 : nSample )';
        grpPred = max( grpScore{ g }, [], 2 );
        scores( sampleIdx, grpPred ) = scores( sampleIdx, grpPred ) + 1;
      end
    otherwise
      fprintf( '\t Error: unknown fusion type: %s\n', conf.fusionPath );
      return;
  end

end % end if isSVMProb


% get cconfusion matrix
[ confusion, meanAcc ] = ScoreToConf( scores( test, : ), imdb.clsLabel( test ) );
% set fusion struct
fusion.scores    = scores;
fusion.confusion = confusion;
fusion.meanAcc   = meanAcc;

fprintf( '\t fusion acc: %.2f %%\n', meanAcc );

% end function GroupFusion


