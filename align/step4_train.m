%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step4_train.m
% Desc: parallel run training and testing 10 processes each has 20 class
% Author: Zhang Kang
% Date: 2013/12/06
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step4_train( chunkID, CHUNK_NUM )

% Step4: Training
tic;
fprintf( '\n Step4: Training ...\n' );


% initial all configuration
initConf;
% temporary model files
conf.chunkNum = CHUNK_NUM;                         % parallel jobs for training
conf.tmpModelPath = cell( 1, conf.chunkNum );
for ii = 1 : conf.chunkNum
  tempFn = sprintf( '-tmpModel%03d.mat', ii );
  conf.tmpModelPath{ ii } = fullfile( conf.outDir, [conf.prefix tempFn] );
end
% setup dataset
setupCUB11;

fprintf( '\n Training chunkID: %d (%.2f %%)', chunkID, 100 * chunkID / conf.chunkNum );

if( ~exist( conf.tmpModelPath{ chunkID } ) )
  
  % load econded features
  if( exist( conf.featPath ) )
    fprintf( '\n Loading encoded features ...\n' );
    load( conf.featPath );
    %% Step 3: training
    fprintf( '\n Training ...\n' );
    
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
    
    
    % train and test (left right flip is not implemented)
    
    numClasses = numel( imdb.clsName );
    train = find( imdb.ttSplit == 1 ) ;
    test = find( imdb.ttSplit == 0 ) ;
    % ??? what is lambda ???
    % SVM parameters
    lambda = 1 / ( conf.svm.C * numel( train ) ) ;
    par = { 'Solver', 'sgd', 'Verbose', ...
      'BiasMultiplier', 1, ...
      'Epsilon', 0.001, ...
      'MaxNumIterations', 100 * numel( train ) } ;
    
    scores = cell(1, numel(imdb.clsName)) ;
    ap = zeros(1, numel(imdb.clsName)) ;
    ap11 = zeros(1, numel(imdb.clsName)) ;
    tmpW = cell(1, numel(imdb.clsName)) ;
    tmpB = cell(1, numel(imdb.clsName)) ;
    % chunck start class and end class
    CHUNCK_SZ = floor( numClasses / conf.chunkNum );
    clsSt = ( chunkID - 1  ) * CHUNCK_SZ + 1;
    clsEd = chunkID * CHUNCK_SZ;
    if( chunkID == conf.chunkNum )
      clsEd = numClasses;
    else
      clsEd = chunkID * CHUNCK_SZ;
    end
    for c = clsSt : clsEd
      fprintf( '\n\t training class: %s (%.2f %%)', ...
        imdb.clsName{ c }, 100 * c / numClasses );
      % one-vs-rest SVM
      y = 2 * ( imdb.clsLabel == c ) - 1 ;
      % perm training set
      perm = randperm( length( train ) );
      [ tmpW{c}, tmpB{c} ] = vl_svmtrain( descrs( :, train( perm ) ), ...
        y( train( perm ) ), lambda, par{ : } ) ;
    end
    % save model
    save( conf.tmpModelPath{ chunkID }, 'tmpW', 'tmpB' ) ;
  else
    fprintf( '\n\t Error: aggregate feat file %s does not exist', conf.featPath );
  end
end

% record time
fprintf( '\n ... Done Step4: Training time: %.2f (s)', toc );

