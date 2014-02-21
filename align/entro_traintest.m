%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: entro_traintest.m
% Desc: use entropy to remove training data
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function entro_traintest()

% Step4: Training and Testing
tic;
fprintf( '\n Step5: Training and Testing ...\n' );

% initial all configuration
initConf;

% setup dataset
switch conf.dataset
  case {'CUB11'}
    setupCUB11;
  case {'STDog'}
    setupSTDog;
end


% load precomputed kernel
if( exist( conf.kernelPath, 'file' ) )
  fprintf( '\n Loading kernel map ...\n' );
  load( conf.kernelPath );
else
  fprintf( '\n\t Error: kernel matrix file %s does not exist', conf.kernelPath );
  return;
end

%% Step 5: training
fprintf( '\n Training and Testing...\n' );

% load entropy 
load( 'entro.mat' ); % --> curEntro

numClasses = numel( imdb.clsName );
train = find( imdb.ttSplit == 1 );
entroTrain = [];
for c = 1 : numClasses
  % select each class half data
  curIdx = intersect( find( imdb.clsLabel == c ), train );
  [ ~, sortIdx ] = sort( curEntro( curIdx ), 'descend' );
  entroTrain = [ entroTrain; curIdx( sortIdx( 1 : ceil( length( curIdx ) / 2 ) ) ) ];
end

test = find( imdb.ttSplit == 0 ) ;

scores = cell( 1, numClasses );
trainScore = cell( 1, numClasses );
ap = zeros( 1, numClasses );
ap11 = zeros( 1, numClasses );

model = cell( 1, numClasses );

% conver all kernel to train and test kernel
numTrain = length( entroTrain );
numTest  = length( test );
kernelTrain = [ ( 1 : numTrain )', ...
  kernel( entroTrain, entroTrain ) ];
kernelTest = [ ( 1 : numTest )', ...
  kernel( test, entroTrain ) ];
clear kernel;

for c = 1 : numClasses
  fprintf( '\n\t training class: %s (%.2f %%)\n', ...
    imdb.clsName{ c }, 100 * c / numClasses );
  % one-vs-rest SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  model{ c } = libsvmtrain( double( y( entroTrain ) ), double( kernelTrain ), ...
    '-c 10 -t 4' ) ;
  % predict on train samples
  [ ~, ~, trainScore{ c } ] = libsvmpredict( double( y( entroTrain ) ), ...
    double( kernelTrain ), model{ c } );
  % predict on test samples
  [ predClass, acc, scores{ c } ] = libsvmpredict( double( y( test ) ), ...
    double( kernelTest ), model{ c } );
  if( isempty( find( predClass == 1 ) ) )
    fprintf( '\n\t Warning: no positive prediction\n' );
  end
  [~,~,info] = vl_pr( y( test ), scores{ c } ) ;
  ap(c) = info.ap ;
  ap11(c) = info.ap_interp_11 ;
  fprintf('\n\t class %s AP %.2f; AP 11 %.2f\n', imdb.clsName{ c }, ...
    ap( c ) * 100, ap11( c ) * 100 ) ;
end

% save model
save( conf.modelPath, 'model' ) ;

fprintf( '\n ... Done Libsvm Training and Testing time: %.2f (s)', toc );


%save results and figures
fprintf( '\n Saving results and figures ...\n' );

% train confusion matrix
trainScore = cat( 2, trainScore{ : } ) ;
[ ~, trainPred ] = max(trainScore, [], 2) ;
trainConf = confusionmat( imdb.clsLabel( entroTrain ), trainPred );
for c = 1 : numClasses
  sumC = sum( trainConf( c , : ) );
  trainConf( c, : ) = trainConf( c, : ) / sumC;
end
fprintf( '\n  train acc: %.2f %%\n', 100 * mean( diag( trainConf ) ) );

% test confusion matrix
scores = cat(2,scores{:}) ;
[~,preds] = max(scores, [], 2) ;
confusion = confusionmat( imdb.clsLabel( test ), preds );
for c = 1 : numClasses
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) / sumC;
end
fprintf( '\n  test acc: %.2f %%\n', 100 * mean( diag( confusion ) ) );

% save result
save( conf.resultPath, ...
  'preds', 'ap', 'ap11', 'scores', 'trainScore', ...
  'trainConf', 'confusion', 'conf' );

% generate figures
meanAccuracy = sprintf('mean accuracy: %.2f %%\n', 100 * mean(diag(confusion)));
mAP = sprintf('mAP: %.2f %%; mAP 11: %.2f %%', mean(ap) * 100, mean(ap11) * 100) ;

figure(1) ; clf ;
imagesc(confusion) ; axis square ;
title([conf.prefix ' - ' meanAccuracy]) ;
vl_printsize(1) ;
print('-dpdf', fullfile(conf.outDir, [ conf.prefix, '-confusion.pdf' ] ) ) ;
%print('-djpeg', fullfile(conf.outDir, 'result-confusion.jpg')) ;

figure(2) ; clf ; bar(ap * 100) ;
title([conf.prefix ' - ' mAP]) ;
ylabel('AP %%') ; xlabel('class') ;
grid on ;
vl_printsize(1) ;
ylim([0 100]) ;
print('-dpdf', fullfile(conf.outDir, [ conf.prefix, '-ap.pdf' ] ) ) ;

fprintf( '\n ... Done\n' );

