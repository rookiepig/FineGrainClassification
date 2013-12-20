%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: runTrainTest.m
% Desc: run train and test
% Author: Zhang Kang
% Date: 2013/12/06
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 1: initial all configuration
initConf;

%% Step 2: setup dataset
setupCUB11;

%% Setep 3: train encoder
if exist( conf.encoderPath, 'file' )
  % load existing encoder
  encoder = load( conf.encoderPath ) ;
else
  numTrain = 5000 ;    % training images for encoding
  encSelTrain = vl_colsubset( transpose( find( imdb.ttSplit == 1 ) ), ...
    numTrain, 'uniform' ) ;
  if( conf.useSegMask )
    % use boudingbox + segment mask
    encoder = TrainEncoder( ...
      fullfile( imdb.imgDir, imdb.imgName( encSelTrain ) ), ...
      imdb.bdBox( encSelTrain, : ), ...
      fullfile( imdb.imgDir, imdb.maskName( encSelTrain ) ) ...
      );
  else
    if( conf.useBoundingBox )
      % only use bounding box
      encoder = TrainEncoder( ...
        fullfile( imdb.imageDir, imdb.imgName{ encSelTrain } ), ...
        imdb.bdBox( encSelTrain, : ) ) ;
    end
  end
  save( conf.encoderPath, '-struct', 'encoder' ) ;
end

%% Step 4: encoding all images
fprintf( '\n Encoding all images ...\n' );
descrs = cell( 1, numel( imdb.imgName ) );
for ii = 1 : numel( imdb.imgName )
  fprintf( '\n\t encoding %s (%.2f %%)', imdb.imgName{ ii }, ...
    100 * ii / numel( imdb.imgName ) );
  if( conf.useSegMask )
    descrs{ ii } = EncodeImg( encoder, ...
      fullfile( imdb.imgDir, imdb.imgName{ ii } ), ...
      imdb.bdBox( ii, : ), ...
      fullfile( imdb.imgDir, imdb.maskName{ ii } ) ) ;
  else
    if( conf.useBoundingBox )
      descrs{ ii } = EncodeImg( encoder, ...
        imdb.bdBox( ii, : ), ...
        fullfile( imdb.imgDir, imdb.imgName{ ii } ) ) ;
    end
  end
end
descrs = cat( 2, descrs{ : } );
fprintf( '\n ... Done\n' );

%% Step 5: training and testing
fpritnf( '\n Training and Testing ...\n' );

% apply kernel maps
switch conf.svm.kernel
  case 'linear'
  case 'hell'
    descrs = sign(descrs) .* sqrt(abs(descrs)) ;
  case 'chi2'
    descrs = vl_homkermap(descrs,1,'kchi2') ;
  otherwise
    assert(false) ;
end
descrs = bsxfun(@times, descrs, 1./sqrt(sum(descrs.^2))) ;


% train and test
if( conf.isLRFlip )
  % !!! test need use LR flip information
  
end

numClasses = numel( imdb.clsName );
train = find( imdb.ttSplit == 1 ) ;
test = find( imdb.ttSplit == 0 ) ;
% ??? what is lambda ???
lambda = 1 / ( conf.svm.C * numel( train ) ) ;
par = { 'Solver', 'sdca', 'Verbose', ...
  'BiasMultiplier', 1, ...
  'Epsilon', 0.001, ...
  'MaxNumIterations', 100 * numel( train ) } ;
scores = cell(1, numel(imdb.clsName)) ;
ap = zeros(1, numel(imdb.clsName)) ;
ap11 = zeros(1, numel(imdb.clsName)) ;
w = cell(1, numel(imdb.clsName)) ;
b = cell(1, numel(imdb.clsName)) ;
for c = 1 : numClasses
  fprintf( '\n\t training class: %s (%.2f %%)', ...
    imdb.clsName{ c }, 100 * c / numClasses );
  % one-vs-rest SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  % perm training set
  perm = randperm( length( train ) );
  [ w{c}, b{c} ] = vl_svmtrain( descrs( :, train( perm ) ), ...
    y( train( perm ) ), lambda, par{ : } ) ;
  scores{c} = w{c}' * descrs + b{c} ;
  
  [~,~,info] = vl_pr( y( test ), scores{c}(test)) ;
  ap(c) = info.ap ;
  ap11(c) = info.ap_interp_11 ;
  fprintf('\n\t class %s AP %.2f; AP 11 %.2f\n', imdb.clsName{ c }, ...
    ap( c ) * 100, ap11( c ) * 100 ) ;
end
scores = cat(1,scores{:}) ;
% save model
save( conf.modelPath, 'w', 'b' ) ;

fprintf( '\n ... Done\n' );




%% Step 5: save results and fiture
fpritnf( '\n Saving results and figures ...\n' );

% confusion matrix (can be computed only if each image has only one label)
[~,preds] = max(scores, [], 1) ;
confusion = zeros(numClasses) ;
for c = 1 : numClasses
  sel = find( imdb.clsLabel == c & imdb.ttSplit == 0 ) ;
  % accumarray() --> useful function
  tmp = accumarray( preds(sel)', 1, [ numClasses 1 ] ) ;
  tmp = tmp / max(sum(tmp),1e-10) ;
  confusion( c, : ) =  tmp( : )' ;
end

save( conf.resultPath, ...
  'scores', 'ap', 'ap11', ...
  'confusion', 'imdb.clsName', 'conf' );

% generate figures
meanAccuracy = sprintf('mean accuracy: %f\n', mean(diag(confusion)));
mAP = sprintf('mAP: %.2f %%; mAP 11: %.2f', mean(ap) * 100, mean(ap11) * 100) ;

figure(1) ; clf ;
imagesc(confusion) ; axis square ;
title([conf.prefix ' - ' meanAccuracy]) ;
vl_printsize(1) ;
print('-dpdf', fullfile(conf.outDir, 'result-confusion.pdf')) ;
print('-djpeg', fullfile(conf.outDir, 'result-confusion.jpg')) ;

figure(2) ; clf ; bar(ap * 100) ;
title([conf.prefix ' - ' mAP]) ;
ylabel('AP %%') ; xlabel('class') ;
grid on ;
vl_printsize(1) ;
ylim([0 100]) ;
print('-dpdf', fullfile(conf.outDir,'result-ap.pdf')) ;
fprintf( '\n ... Done\n' );

