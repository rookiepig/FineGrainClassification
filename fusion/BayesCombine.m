function [ prob ] = BayesCombine( conf, imdb, curGrp, curModel )
%% BayesCombine
%  Desc: combine cluster probability and class probability
%  In: 
%    conf, imdb -- basic variables
%    curGrp, curModel -- (struct) current group info
%  Out:
%    prob -- (nSample * nClass) probability for each class
%%
PrintTab();fprintf( 'function: %s\n', mfilename );

nSample = length( imdb.clsLabel );
nClass  = max( imdb.clsLabel );

% bayes output for each group
PrintTab();fprintf( 'get bayes prob\n' );
% init current group final probability
prob = zeros( nSample, nClass );
for t = 1 : curGrp.nCluster
  grpCls = curGrp.cluster{ t };
  % get cluster prior prob
  clusterProb = curGrp.clusterProb( :, t );
  for c = 1 : length( grpCls )
    curCls = grpCls( c );
    % get class likelihood prob
    clsProb = curModel.probFeat( :, curCls );
    % bayes rule to get final class prob
    prob( :, curCls ) = prob( :, curCls ) + ...
      clsProb .* clusterProb;
  end % end for each class
end % end for each cluster

% end function BayesCombine

