%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step5_libsvm_traintest.m
% Desc: serial training and testing using precomputed kernel map
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step5_libsvm_traintest()

% Step4: Training and Testing
tic;
fprintf( '\n Step5: Training and Testing ...\n' );

% initial all configuration
initConf;

% setup dataset
setupCUB11;


% load precomputed kernel
if( exist( conf.kernelPath, 'file' ) )
  fprintf( '\n Loading kernel map ...\n' );
  load( conf.kernelPath );
else
  fprintf( '\n\t Error: kernel matrix file %s does not exist', conf.kernelPath );
end

%% Step 5: training
fprintf( '\n Training and Testing...\n' );
% train and test (left right flip is not implemented)

numClasses = numel( imdb.clsName );
train = ( imdb.ttSplit == 1 );
test = ( imdb.ttSplit == 0 ) ;

scores = cell( 1, numClasses );
ap = zeros( 1, numClasses );
ap11 = zeros( 1, numClasses );

model = cell( 1, numClasses );

for c = 1 : numClasses
  fprintf( '\n\t training class: %s (%.2f %%)\n', ...
    imdb.clsName{ c }, 100 * c / numClasses );
  % one-vs-rest SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  model{ c } = libsvmtrain( double( y( train ) ), double( kernelTrain ), ...
    '-c 10 -t 4' ) ;
  [predClass, acc, scores{ c } ] = libsvmpredict( double( y( test ) ), ...
    double( kernelTest ), model{ c } );
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

scores = cat(2,scores{:}) ;
% confusion matrix
[~,preds] = max(scores, [], 2) ;
confusion = confusionmat( imdb.clsLabel( test ), preds );
for c = 1 : numClasses
  sumC = sum( confusion( c , : ) );
  confusion( c, : ) = confusion( c, : ) / sumC;
end

% save result
save( conf.resultPath, ...
  'preds', 'ap', 'ap11', 'scores', ...
  'confusion', 'conf' );

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

