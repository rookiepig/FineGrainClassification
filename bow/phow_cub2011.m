function phow_cub2011()

run( '~/vlfeat/toolbox/vl_setup' );

% load CUB data
fprintf( 1, 'Load data...\n' );
load( '../../in/CUB2011/cub2011.mat' );
% open matlab pool (clustter maximum is 8)
% matlabpool(8);

conf.calDir = '../../in/CUB2011/images' ;
conf.dataDir = 'data/' ;
conf.numClasses = 200;
conf.numWords = 2048; % codebook member
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;
conf.svm.solver = 'sdca' ;

%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
% add opponent color
conf.phowOpts = {'Step', 3 } ;
conf.clobber = false ;
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;


conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
% add feature map path
conf.featPath = fullfile(conf.dataDir, [conf.prefix '-feat.mat']);

conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                                  Setup CUB 2011 data
% --------------------------------------------------------------------
fprintf( 1, 'Setup CUB 2011 data...\n' );

classes = clsName( :, 2 );         % class names

images = imgName( :, 2 );          % image file name (no directory)
imageClass = clsLabel( :, 2 );     % image class label

selTrain = find( ttSplit( :, 2 ) == 1 ); % training index
selTest  = find( ttSplit( :, 2 ) == 0 ); % test index

model.classes = classes;
model.phowOpts = conf.phowOpts;
model.numSpatialX = conf.numSpatialX;
model.numSpatialY = conf.numSpatialY;
model.quantizer = conf.quantizer;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;

% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------
tic;

fprintf( 1, 'Train vocabulary...\n' );

if ~exist(conf.vocabPath) || conf.clobber

  % Get some PHOW descriptors to train the dictionary
  selTrainFeats = ( vl_colsubset(selTrain', 200) )';
  descrs = {} ;
  for ii = 1:length(selTrainFeats)
    fprintf( 1, '\n\t %d train vocab', ii );
    im = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
    im = standarizeImage(im, bdBox( selTrainFeats( ii ), 2 : 5 )  ) ;
    [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
  end

  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;

  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  save(conf.vocabPath, 'vocab') ;
else
  load(conf.vocabPath) ;
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end

vocabTime = toc;

fprintf( 1, 'Time for Train Vocabulary: %.2f\n', vocabTime );

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------
tic;
fprintf( 1, 'Compute spatial histograms...\n' );

if ~exist(conf.histPath) || conf.clobber
  hists = {} ;
  for ii = 1:length(images)
    fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
    im = imread(fullfile(conf.calDir, images{ii})) ;
    hists{ii} = getImageDescriptor(model, im, bdBox( ii, 2 : 5 ) );
  end

  hists = cat(2, hists{:}) ;
  save(conf.histPath, 'hists') ;
else
  load(conf.histPath) ;
end

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------
fprintf( 1, 'Compute feature map...\n' );

if ~exist(conf.featPath) || conf.clobber
    psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
    save(conf.featPath, 'psix', '-v7.3');
else
    load(conf.featPath);
end
featTime = toc;
fprintf( 1, 'Time for Compute feature map: %.2f\n', featTime );
% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------
tic;

fprintf( 1, 'Train SVM...\n' );

if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
      w = [] ;
      for ci = 1:length(classes)
        perm = randperm(length(selTrain)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (imageClass(selTrain) == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain)', ...
                  sparse(double(psix(:,selTrain))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end

trainTime = toc;

fprintf( 1, 'Time for training: %.2f\n', trainTime );

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------
tic;

fprintf( 1, 'Test SVM...\n' );

% Estimate the class of the test images
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;
imageEstClass = imageEstClass';
% debug
fprintf( 1, 'psix size: %d\n', size(psix) );
fprintf( 1, 'imageEstClass size: %d\n', size( imageEstClass ) );
fprintf( 1, 'imageClass size: %d\n', size( imageClass ) );
save( 'imgEstCls.mat', 'imageEstClass' );
save( 'imgCls.mat', 'imageClass' );
% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
              imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;
% compute each class test num
idx = sub2ind([length(classes), 1 ], ...
              imageClass(selTest)) ;
testClsNum  = zeros(length(classes), 1) ;
testClsNum = vl_binsum(testClsNum, ones(size(idx)), idx) ;


% Plots
figure(1) ; clf;
%subplot(1,2,1) ;
%imagesc(scores(:,[selTrain;  selTest])) ; title('Scores') ;
%set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
%subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
              100 * mean(diag(confus)./testClsNum ))) ;
print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf', 'testClsNum' ) ;

testTIme = toc;

fprintf( 1, 'Time for testing: %.2f\n', testTIme );

% close matlabpool
% matlabpool( 'close' );

end

% -------------------------------------------------------------------------
function im = standarizeImage(im, bdBox)
% -------------------------------------------------------------------------
% resize image, so that maximum dimmension does not exceed 300
im = imcrop( im, bdBox ); % crop image to bounding box
im = im2single(im) ;
wid = size( im, 2 );
hei = size( im, 1 );
if ( hei > 300 ) || ( wid > 300 )
    if( hei > wid )
        im = imresize( im, [ 300, NaN ] );
    else
        im = imresize( im, [ NaN, 300 ] );
    end
end

end
% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im, bdBox )
% -------------------------------------------------------------------------

im = standarizeImage(im, bdBox ) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;
end
% -------------------------------------------------------------------------
function [className, score] = classify(model, im, bdBox )
% -------------------------------------------------------------------------

hist = getImageDescriptor(model, im, bdBox ) ;
psix = vl_homkermap(hist, 1, 'kchi2', 'period', .7) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;
end
