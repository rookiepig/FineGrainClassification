%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step5_test.m
% Desc: aggregate parallel training results and testing
% Author: Zhang Kang
% Date: 2013/12/06
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step5_test( CHUNK_NUM )

% Step5: Testing
tic;
fprintf( '\n Step5: Testing ...\n' );


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

% load econded features
if( exist( conf.featPath ) )
  fprintf( '\n Loading aggregate features ...\n' );
  load( conf.featPath );
else
  fprintf( '\n\t Error: aggregate feature file %s does not exist', conf.featPath );
end

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


% test (left right flip is not implemented)


numClasses = numel( imdb.clsName );
train = find( imdb.ttSplit == 1 ) ;
test = find( imdb.ttSplit == 0 ) ;

scores = cell(1, numel(imdb.clsName)) ;
ap = zeros(1, numel(imdb.clsName)) ;
ap11 = zeros(1, numel(imdb.clsName)) ;

% aggregate all model files
if( exist( conf.modelPath ) )
  fprintf( '\n\t load model from file: %s', conf.modelPath );
  load( conf.modelPath );
else
  
  w = cell(1, numel(imdb.clsName)) ;
  b = cell(1, numel(imdb.clsName)) ;
  fprintf( '\n\t aggregate each training chunk' );
  CHUNCK_SZ = floor( numClasses / conf.chunkNum );
  for chunkID = 1 : conf.chunkNum
    if( exist( conf.tmpModelPath{ chunkID }, 'file' ) )
      fprintf( '\n\t load temp model file: %s', conf.tmpModelPath{ chunkID } );
      load( conf.tmpModelPath{ chunkID } );
      % chunck start class and end class
      clsSt = ( chunkID - 1  ) * CHUNCK_SZ + 1;
      clsEd = chunkID * CHUNCK_SZ;
      if( chunkID == conf.chunkNum )
        clsEd = numClasses;
      else
        clsEd = chunkID * CHUNCK_SZ;
      end
      w( clsSt : clsEd ) = tmpW( clsSt : clsEd );
      b( clsSt : clsEd ) = tmpB( clsSt : clsEd );
      
      % delete tmp model file
      clear tmpW, tmpB;
      delete( conf.tmpModelPath{ chunkID } );
    else
      fprintf( 2, 'Error: tmp model file %s does not exist\n', ...
        conf.tmpModelPath{ chunkID } );
      exit;
    end
    
  end
  % save model
  save( conf.modelPath, 'w', 'b' ) ;
end


for c = 1 : numClasses
  fprintf( '\n\t testing class: %s (%.2f %%)', ...
    imdb.clsName{ c }, 100 * c / numClasses );
  % one-vs-rest SVM
  y = 2 * ( imdb.clsLabel == c ) - 1 ;
  scores{c} = w{c}' * descrs + b{c} ;
  [~,~,info] = vl_pr( y( test ), scores{c}(test)') ;
  ap(c) = info.ap ;
  ap11(c) = info.ap_interp_11 ;
  fprintf('\n\t class %s AP %.2f; AP 11 %.2f\n', imdb.clsName{ c }, ...
    ap( c ) * 100, ap11( c ) * 100 ) ;
end

fprintf( '\n ... Done\n' );

fprintf( '\n Step5: testing time: %.2f (s)', toc );

%% Step 5: save results and figures
fprintf( '\n Saving results and figures ...\n' );

% confusion matrix
scores = cat(1,scores{:}) ;
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
  'confusion', 'conf' );

% generate figures
meanAccuracy = sprintf('mean accuracy: %f\n', mean(diag(confusion)));
mAP = sprintf('mAP: %.2f %%; mAP 11: %.2f', mean(ap) * 100, mean(ap11) * 100) ;

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

