function [confusion, meanAcc] = ScoreToConf( scores, label )
%% ScoreToConf
%  Desc: get confusion matrix from one-vs-all scores
%  In: 
%    scores -- (nSample * nClass) one-vs-all scores
%    label  -- ground truth label
%  Out:
%    confusion -- (nClass * nClass) confusion matrix
%    meanAcc   -- mean accuracy
%%

% PrintTab();fprintf( 'function: %s\n', mfilename );

% confusion matrix
nClass = max( label );
[ ~, preds ] = max( scores, [], 2 ) ;
confusion = confusionmat( label, preds );
for c = 1 : nClass
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) ./ max( sumC, 1e-12 );
end

meanAcc = 100 * mean(diag(confusion));

% fprintf( '\t Mean accuracy: %.2f %%\n', meanAcc );

% end function ScoreToConf


