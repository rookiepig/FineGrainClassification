function fusion = GroupFusion( conf, imdb, grpInfo, grpModel )
%% GroupFusion
%  Desc: calculate each group result and fuse
%  In: 
%    conf, imdb -- basic variables
%    grpInfo  -- (struct) clustering info of all groups
%    grpModel -- (struct) all group models
%  Out:
%    fusion -- (struct) fusion model and results
%%

PrintTab();fprintf( 'function: %s\n', mfilename );


nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );
test = find( imdb.ttSplit == 0 );
testLab = imdb.clsLabel( test );

% each group's result
grpProb = cell( 1, conf.nGroup );
grpConf = cell( 1, conf.nGroup );
grpAcc  = zeros( 1, conf.nGroup );

% final fusion scores
scores  = zeros( nSample, nClass);

% if( conf.useClusterPrior )
%   % set non-cluster scores to negative infinite
%   testNum = length( test );
%   nClass = max( imdb.clsLabel );
%   for g = 1 : conf.nGroup
%     cTc = grpInfo{ g }.clsToCluster( test );
%     % set other cluster score to minimum
%     for t = 1 : testNum
%       clsIdx = grpInfo{ g }.cluster{ cTc( t ) };
%       clsIdx = setdiff( ( 1 : nClass )', clsIdx );
%       grpModel{ g }.scores( t, clsIdx ) = -1e10;
%     end
%   end
% end

%% calculate each group's performance

PrintTab();fprintf( '\t calculate each group performance\n' );
PrintTab();fprintf( '\t svm score map type: %s\n', conf.mapType );
fusion.mapType = conf.mapType;
switch conf.mapType
  case 'reg'
    % no prob combine --> just use class probFeat
    for g = 1 : conf.nGroup
      PrintTab();fprintf( '\t group %d\n', g );
      % get current group confusion and accuracy
      [ grpConf{ g }, grpAcc( g ) ] = ...
        ScoreToConf( grpModel{ g }.probFeat( test, : ), testLab );
      % set grpProb for final fusion
      grpProb{ g } = grpModel{ g }.probFeat;
      PrintTab();fprintf( '\t current acc: %.2f %%\n', grpAcc( g ) );
    end
  case 'softmax'
    for g = 1 : conf.nGroup
      PrintTab();fprintf( '\t group %d\n', g );
      % get current group confusion and accuracy
      [ grpConf{ g }, grpAcc( g ) ] = ...
        ScoreToConf( grpModel{ g }.bayesProb( test, : ), testLab );
      % set grpProb for final fusion
      grpProb{ g } = grpModel{ g }.bayesProb;
      PrintTab();fprintf( '\t current acc: %.2f %%\n', grpAcc( g ) );
    end % end for each group
  otherwise
    PrintTab();fprintf( 'Error: unknown map type: %s\n', conf.mapType );
end
% save each group result
fusion.grpProb = grpProb;
fusion.grpConf = grpConf;
fusion.grpAcc  = grpAcc;

%% group fusion

PrintTab();fprintf( '\t fusion type: %s\n', conf.fusionType );
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
        scores = scores + full( sparse( sampleIdx, grpPred, 1, nSample, nClass ) );
    end
  case 'reg'
    allFeat = [];
    for g = 1 : conf.nGroup
      PrintTab();fprintf( '\t group %d\n', g );
      % concatante all grpProb and re mapping using regression
      allFeat = [ allFeat grpProb{ g } ];
    end
    % get final mapping
    scores = TrainMapReg( conf, imdb, allFeat, imdb.clsLabel );
  otherwise
    PrintTab();fprintf( '\t Error: unknown fusion type: %s\n', conf.fusionType );
end


% get fusion cconfusion matrix
[ confusion, meanAcc ] = ScoreToConf( scores( test, : ), testLab );
% set fusion struct
fusion.scores    = scores;
fusion.confusion = confusion;
fusion.meanAcc   = meanAcc;

PrintTab();fprintf( '\t fusion acc: %.2f %%\n', meanAcc );

% end function GroupFusion


