%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step4_libsvm_traintest.m
% Desc: serial training and testing using precomputed kernel map
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step4_libsvm_traintest()

% Step4: Training and Testing 
tic;
fprintf( '\n Step4: Training and Testing ...\n' );

% initial all configuration
initConf;

% setup dataset
setupCUB11;


% load econded features
if( exist( conf.featPath ) )
    fprintf( '\n Loading kernel map ...\n' );
    load( conf.featPath );
else
    fprintf( '\n\t Error: kernel matrix file %s does not exist', conf.featPath );
    exit;
end

%% Step 4: training
fprintf( '\n Training and Testing...\n' );
% train and test (left right flip is not implemented)
    
numClasses = numel( imdb.clsName );
train = find( imdb.ttSplit == 1 ) ;
test = find( imdb.ttSplit == 0 ) ;

fprintf( '\n\t training all classes' );

% perm training set (kernel matrix do not perm)
% perm = randperm( length( train ) );
model = svmtrain( imdb.clsLabel( train ), kernelTrain, ...
    '-t 4' ) ;
[predClass, acc, decVals] = svmpredict( imdb.clsLabel( test ), ...
    kernelTest, model );

% save model
save( conf.modelPath, 'model' ) ;

fprintf( '\n ... Done Step4: Libsvm Training and Testing time: %.2f (s)', toc );


%% Step 4: save results and figures
fprintf( '\n Saving results and figures ...\n' );

%# confusion matrix
confusion = confusionmat( imdb.clsLabel( test ), predClass );

% save result
save( conf.resultPath, ...
    'predClass', 'acc', 'decVals', ...
    'confusion', 'conf' );

% generate figures
meanAccuracy = sprintf('mean accuracy: %f\n', mean(diag(confusion)));
figure(1) ; clf ;
imagesc(confusion) ; axis square ;
title([conf.prefix ' - ' meanAccuracy]) ;
vl_printsize(1) ;
print('-dpdf', fullfile(conf.outDir, [ conf.prefix, '-confusion.pdf' ] ) ) ;

fprintf( '\n ... Done\n' );

