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
switch conf.fusionType
  case 'average'
      % average SVM scores
      fprintf( '\t Average Fusion\n' );
      scores = zeros( size( grpModel{ 1 }.scores ) );
      for g = 1 : conf.nGroup
        scores = scores + grpModel{ g }.scores;
      end
      scores = scores ./ conf.nGroup;
  otherwise
    fprintf( '\t Error: unknown fusion type: %s\n', conf.fusionPath );
    return;
end

[ confusion, ~ ] = ScoreToConf( scores, imdb.clsLabel( test ) );

% set fusion struct
fusion.scores = scores;
fusion.confusion = confusion;


% end function GroupFusion


