%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: genConfusion.m
% Desc: get confusion matrix for Group Fusion
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if( exist( 'curModel', 'var' ) )
  clear curModel;
end

load( 'cache\oracle-tmpModel008.mat' );

% get confusion matrix
fprintf( 'Get confusion matrix\n' );
numClasses = numel( imdb.clsName );
numSample  = numel( imdb.clsLabel );
train = find( imdb.ttSplit == 1 );
test = find( imdb.ttSplit == 0 ) ;

scores = cat(2,curModel.scores{:}) ;
% confusion matrix
[~,preds] = max(scores, [], 2) ;
confusion = confusionmat( imdb.clsLabel( test ), preds );
for c = 1 : numClasses
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) / sumC;
end
meanAccuracy = sprintf('mean accuracy: %.2f %%\n', 100 * mean(diag(confusion)));
figure(1) ; clf ;
imagesc(confusion) ; axis square ;
title(meanAccuracy) ;

%print('-djpeg', fullfile(conf.outDir, 'result-confusion.jpg')) ;

