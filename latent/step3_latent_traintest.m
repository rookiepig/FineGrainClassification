%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step3_latent_traintest.m
% Desc: latent training and testing
% Author: Zhang Kang
% Date: 2014/03/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function step3_latent_traintest( JOB_NUM )

JOB_NUM = 60;

% Step3: latent training and testing
allTime = tic;
fprintf( '\n Step3: Latent training and testing...\n' );

% initial all configuration
initConf;
% setup dataset
switch conf.dataset
  case {'CUB11'}
    setupCUB11;
  case {'STDog'}
    setupSTDog;
end
nSample = length( imdb.clsLabel );

% temporary encoding files
conf.cacheDir = [ 'cache/' conf.dataset ];    % cache dir
conf.jobNum = JOB_NUM;
for ii = 1 : conf.jobNum
  tempFn = sprintf( '-tmpDescrs%03d.mat', ii );
  conf.tmpDescrsPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end

% load econded features to feature matrix
if( ~exist( 'allFeat', 'var' ) )
  allFeat = cell( 1, conf.jobNum );
  for ii = 1 : conf.jobNum
    if( exist( conf.tmpDescrsPath{ ii }, 'file' ) ) % descrs file exist
      fprintf( '   load encoding feature: %03d (%.2f %%) ... \n', ...
        ii, 100 * ii / conf.jobNum  );
      % split image according to ii and jobNum
      ttImgNum = numel( imdb.imgName );
      jobSz = floor( ttImgNum / conf.jobNum );
      jobSt = ( ii - 1 ) * jobSz + 1;
      if( ii == conf.jobNum )
        jobEd = ttImgNum;
      else
        jobEd = ii * jobSz;
      end
      % load current job des
      load( conf.tmpDescrsPath{ ii } );
      jobDes = cat( 2, jobDes{ : } );
      allFeat{ ii } = sparse( double( jobDes' ) );
    end
  end % end for each job
  fprintf( 'convert allFealt...\n' );
  allFeat = cat( 1, allFeat{ : } );
  fprintf( 'save all feat..\n' );
  save( 'tmp/allFeat.mat', 'allFeat', '-v7.3' );
end
fprintf( '\n allFeat size: %d x %d\n', size( allFeat, 1 ), size( allFeat, 2 ) );

% temporary iteration files
conf.tmpIterPath = cell( 1, conf.iterNum );
for ii = 1 : conf.iterNum
  tempFn = sprintf( '-tmpIter%03d.mat', ii );
  conf.tmpIterPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end

% latent training and testing
VIEW_NUM   = size( conf.viewType, 1 );
% TYPE_NUM = size( conf.viewType, 2 );
nClass     = max( imdb.clsLabel );
train      = find( imdb.ttSplit == 1 );
test       = find( imdb.ttSplit == 0 );
testLab    = imdb.clsLabel( test, : );

% init current view to viewType( 1 )
curView = ones( nSample, 1 );
preView = curView;
% iter svm model
model = cell( 1, conf.iterNum );
result = cell( 1, conf.iterNum );

svmOpt = sprintf( '-c %f', conf.svm.C );

for iter = 1 : conf.iterNum
  fprintf( '\n  start iter %d (%.2f %%)\n', iter, 100 * iter / conf.iterNum );
  iterTime = tic;
  model{ iter }.svm = cell( 1, nClass );
  posFeat = cell( 1, VIEW_NUM );
  % update each positive class and model
  fprintf( '  training..\n' );
  for c = 1 : nClass
    fprintf('    train class %d (%.2f %%)\n', c, 100 * c / nClass );
    y = 2 * ( imdb.clsLabel == c ) - 1;
    posTrain = intersect( find( y == 1 ), train );
    y = double( y );
    model{ iter }.svm{ c } = liblineartrain( y( train ), ...
      allFeat( train, : ), svmOpt );
    % generate all views positive feat and predict
    viewScore = zeros( length( posTrain ), VIEW_NUM );
    for v = 1 : VIEW_NUM
      posPreview   = curView( posTrain );
      posCurview   = v * ones( size( posTrain ) );
      posFeat{ v } = UpdateFeat( allFeat( posTrain, : ), posPreview, posCurview );
      [ pred, acc, viewScore( : , v ) ] = liblinearpredict( y( posTrain ), ...
        posFeat{ v }, model{ iter }.svm{ c } );
    end
    % update pos view and pos feat
    preView( posTrain, : ) = curView( posTrain, : );
    [ maxScore, curView( posTrain, : ) ] = max( viewScore, [], 2 );
    allFeat( posTrain, : ) = UpdateFeat( allFeat( posTrain, : ),  ...
      preView( posTrain, : ), curView( posTrain, : ) );
  end % end for class

  % test and save iter results
  fprintf( '  testing...\n' );
  result{ iter }.testScore = zeros( length( test ), nClass );
  testFeat = cell( 1, VIEW_NUM );
  for c = 1 : nClass
    fprintf('    test class %d (%.2f %%)\n', c, 100 * c / nClass );
    y = ( 2 * imdb.clsLabel == c ) - 1;
    y = double( y );
    for v = 1 : VIEW_NUM
      % generate test view feat
      testPreview   = preView( test );
      testCurview   = v * ones( size( test ) );
      testFeat{ v } = UpdateFeat( allFeat( test, : ), testPreview, testCurview );
      [ pred, acc, viewScore( : , v ) ] = liblinearpredict( y( posTrain ), ...
        posFeat{ v }, model{ iter }.svm{ c } );
    end
    % update test view and test feat
    preView( test, : ) = curView( test, : );
    [ result{ iter }.testScore( :, c ), curView( test, : ) ] = ...
      max( viewScore, [], 2 );
    allFeat( test, : ) = UpdateFeat( allFeat( test, :), preView( test, : ), ...
      curView( test, : ) );
  end
  % get conf and acc
  [ result{ iter }.conf, result{ iter }.testAcc ] = ScoreToConf( ...
    result{ iter }.testScore, testLab );
  fprintf( '\n iter %d -- test acc: %.2f %%\n', result{ iter }.testAcc );
  
  fprintf( '    save current iter model and results\n' );
  save( conf.tmpIterPath{ ii }, 'model', 'result', 'curView', 'preView' );

  fprintf( '\n  iter %d -- time: %.2f (s)\n', toc( iterTime ) );
end % end for iter

fprintf( '\n ...Done Latent training&testing time: %.2f (s)', toc( allTime ) );
