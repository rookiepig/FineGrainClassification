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
if( conf.lite ) 
  fprintf( 'setup toy data\n' );
  setupToy;
else
  switch conf.dataset
    case {'CUB11'}
      setupCUB11;
    case {'STDog'}
      setupSTDog;
  end
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
  % allFeat = cell( 1, conf.jobNum );
  allFeat = zeros( 393216, nSample, 'single' );
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
      allFeat( :, jobSt : jobEd ) = jobDes;
      % allFeat{ ii } = sparse( double( jobDes' ) );
    end
  end % end for each job
  clear jobDes;
  % fprintf( 'convert allFealt..');
  % fprintf( 'save all feat..\n' );
%     fprintf( 'save all feat to %s\n', conf.allFeatPath );
%     save( conf.allFeatPath, 'allFeat', '-v7.3' );
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

svmOpt = sprintf( '-c %f', conf.svm.C );
if( conf.lite )
  visToy;
  title( 'Init allFeat\n' );
end

for iter = 1 : conf.iterNum
  iterTime = tic;
  fprintf( '\n  start iter %d (%.2f %%)\n', iter, 100 * iter / conf.iterNum );
  if( exist( conf.tmpIterPath{ iter }, 'file' ) )
    fprintf( 'load cur iter from file %s\n', conf.tmpIterPath{ iter } );
    load( conf.tmpIterPath{ iter } );
  else
    model.w = cell( 1, nClass );
    model.b = cell( 1, nClass );
    if( conf.lite )
      % train  original model
      for c = 1 : nClass
        fprintf('    train class %d (%.2f %%)\n', c, 100 * c / nClass );
        y = 2 * ( imdb.clsLabel == c ) - 1;
        % svm weight
        wgt = ones( size( y ) );
        allPosTrain = intersect( find( y == 1 ), train );
        wgt( allPosTrain ) = ( nClass - 1 );
        y = double( y );
        % train final model
        fprintf( '      new model with updated view\n' );
        lambda = 1 / ( conf.svm.C * length( train ) );
        [ model.w{ c }, model.b{ c }, curInfo ] = ...
          vl_svmtrain( allFeat( :, train ), y( train ), lambda, ...
            'Solver', conf.svm.solver, ...
            'MaxNumIterations', 50 / lambda, ...
            'BiasMultiplier', conf.svm.biasMultiplier, ...
            'Epsilon', 1e-3, ...
            'Weights', wgt( train ) ...
          );
      end % end for class
      % test original score
        % test and save iter results
        preView( test, : ) = ones( length( test ), 1 );
        curView( test, : ) = ones( length( test ), 1 );
        fprintf( '  testing...\n' );
        result.testScore = zeros( length( test ), nClass );
        for c = 1 : nClass
          fprintf('    test class %d (%.2f %%)\n', c, 100 * c / nClass );
          y = ( 2 * imdb.clsLabel == c ) - 1;
          y = double( y );
          result.testScore( :, c ) = ( model.w{ c }' * allFeat( :, test ) + ...
            model.b{ c } )';
        end
        clear testFeat;

        % get conf and acc
        [ result.conf, result.testAcc ] = ScoreToConf( ...
          result.testScore, testLab );
        fprintf( '\n iter %d -- test acc: %.2f %%\n', iter, result.testAcc );
    end
    % update each positive class and model
    fprintf( '  training..\n' );
    
    for c = 1 : nClass
      fprintf('    train class %d (%.2f %%)\n', c, 100 * c / nClass );
      y = 2 * ( imdb.clsLabel == c ) - 1;
      % svm weight
      wgt = ones( size( y ) );
      allPosTrain = intersect( find( y == 1 ), train );
      wgt( allPosTrain ) = ( nClass - 1 );
      y = double( y );
      % two-fold split to update view
      spPart  = ceil( length( allPosTrain ) / conf.latentFOLD );
      allPerm = randperm( length( allPosTrain ) );
      for f = 1 : conf.latentFOLD
        fprintf( '      fold %d\n', f );
        fSt = ( f - 1 ) * spPart + 1;
        fEd = min( f * spPart, length( allPosTrain ) );
        posTrain = allPosTrain( allPerm( fSt : fEd ) );
        posTest  = setdiff( allPosTrain, posTrain );
        curTrain = setdiff( train, posTest );
        % model{ iter }.svm{ c } = liblineartrain( y( train ), ...
        %   allFeat( train, : ), svmOpt );
        %  train on current fold
        lambda = 1 / ( conf.svm.C * length( curTrain ) );
        [ model.w{ c }, model.b{ c }, curInfo ] = ...
          vl_svmtrain( allFeat( :, curTrain ), y( curTrain ), lambda, ...
            'Solver', conf.svm.solver, ...
            'MaxNumIterations', 50 / lambda, ...
            'BiasMultiplier', conf.svm.biasMultiplier, ...
            'Epsilon', 1e-3, ...
            'Weights', wgt( curTrain ) ...
          );
        % generate all view pos test feat and predict
        viewScore = zeros( length( posTest ), VIEW_NUM );
        for v = 1 : VIEW_NUM
          posPreview   = curView( posTest );
          posCurview   = v * ones( size( posTest ) );
          posFeat = UpdateFeat( allFeat( :, posTest ), posPreview, posCurview );
          viewScore( : , v ) = ( model.w{ c }' * posFeat + ...
            model.b{ c } )';
        end
        % update pos view and pos feat
        preView( posTest, : ) = curView( posTest, : );
        [ maxScore, curView( posTest, : ) ] = max( viewScore, [], 2 );
        allFeat( :, posTest ) = UpdateFeat( allFeat( :, posTest ),  ...
          preView( posTest, : ), curView( posTest, : ) );
        if( conf.lite )
          visToy;
          title( sprintf( 'Class %d -- Fold %d\n', c, f ) );
        end
      end % end for each fold
      % train final model
      fprintf( '      new model with updated view\n' );
      lambda = 1 / ( conf.svm.C * length( train ) );
      [ model.w{ c }, model.b{ c }, curInfo ] = ...
        vl_svmtrain( allFeat( :, train ), y( train ), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50 / lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3, ...
          'Weights', wgt( train ) ...
        );
    end % end for class
    % save model
    fprintf( '    save current iter model\n' );
    if( ~conf.lite )
      save( conf.tmpIterPath{ ii }, 'model', 'curView', 'preView'  );
    end
    clear posFeat;
  end % if file

  % test and save iter results
  if( conf.oneViewTest ) 
    % single view test
    result.testScore = zeros( length( test ), nClass );
    for c = 1 : nClass
      fprintf('    test class %d (%.2f %%)\n', c, 100 * c / nClass );
      result.testScore( :, c ) = ( model.w{ c }' * allFeat( :, test ) + ...
        model.b{ c } )';
    end
  else
    % multi view test   
    allFeat = allFeat( :, test );   % save memory
    preView( test, : ) = ones( length( test ), 1 );
    curView( test, : ) = ones( length( test ), 1 );
    fprintf( '  testing...\n' );
    result.testScore = zeros( length( test ), nClass );
    for c = 1 : nClass
      fprintf('    test class %d (%.2f %%)\n', c, 100 * c / nClass );
      y = ( 2 * imdb.clsLabel == c ) - 1;
      y = double( y );

      viewScore = zeros( length( test ), VIEW_NUM ) - 100;
      for v = 1 : VIEW_NUM
        % generate test view feat
        testPreview   = preView( test );
        testCurview   = v * ones( size( test ) );
        % testFeat = UpdateFeat( allFeat( :, test ), testPreview, testCurview );
        % viewScore( : , v ) = ( model.w{ c }' * testFeat + ...
        %   model.b{ c } )';
        testFeat = UpdateFeat( allFeat, testPreview, testCurview );
        viewScore( : , v ) = ( model.w{ c }' * testFeat + ...
          model.b{ c } )';
      end
      % update test view and test feat
      [ result.testScore( :, c ), curView( test, : ) ] = ...
        max( viewScore, [], 2 );
      % allFeat( :, test ) = UpdateFeat( allFeat( :, test ), preView( test, : ), ...
      %   curView( test, : ) );
      allFeat = UpdateFeat( allFeat, preView( test, : ), ...
        curView( test, : ) );
      preView( test, : ) = curView( test, : );
    end
  end
  clear testFeat;
  
  % get conf and acc
  [ result.conf, result.testAcc ] = ScoreToConf( ...
    result.testScore, testLab );
  fprintf( '\n iter %d -- test acc: %.2f %%\n', iter, result.testAcc );
  
  if( ~conf.lite )
    fprintf( '    save current iter model and results\n' );
    save( conf.tmpIterPath{ ii }, 'model', 'result', 'curView', 'preView' );
  end

  fprintf( '\n  iter %d -- time: %.2f (s)\n', iter, toc( iterTime ) );
end % end for iter

fprintf( '\n ...Done Latent training&testing time: %.2f (s)', toc( allTime ) );
