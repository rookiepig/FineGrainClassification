%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step4_libsvm_aggre.m
% Desc: aggregate kernel maps
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step4_libsvm_aggre( JOB_NUM )

% Step3: aggregate features
tic;
fprintf( '\n Step4: Aggregate Precomputed Kernel...\n' );


% initial all configuration
initConf;
% temporary encoding files
conf.cacheDir = [ 'cache/' conf.dataset ];            % cache dir for temp files
conf.jobNum = JOB_NUM;
conf.tmpKernelPath = cell( 1, conf.jobNum );
for ii = 1 : conf.jobNum
  tempFn = sprintf( '-tmpKernel%03d.mat', ii );
  conf.tmpKernelPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end

% setup dataset
switch conf.dataset
  case {'CUB11'}
    setupCUB11;
  case {'STDog'}
    setupSTDog;
end


% load econded features
if( exist( conf.kernelPath, 'file' ) )
  fprintf( '\n\t precompute kernel file exist: %s', conf.kernelPath );
else
  ttImgNum = numel( imdb.imgName );
  kernel = zeros( ttImgNum, ttImgNum );
  jobSz = floor( ttImgNum / conf.jobNum );
  
  allExist = true;

  for jobID = 1 : conf.jobNum
    if( exist( conf.tmpKernelPath{ jobID }, 'file' ) )
      % load job kernel
      fprintf( '\n\t load kernel file %s (%.2f %%)', ...
        conf.tmpKernelPath{ jobID }, 100 * jobID / conf.jobNum );
      load( conf.tmpKernelPath{ jobID } );
      rowSt = ( jobID - 1 ) * jobSz + 1;
      if( jobID == conf.jobNum )
        rowEd = ttImgNum;
      else
        rowEd = jobID * jobSz;
      end
      kernel( rowSt : rowEd, : ) = jobKernel;
    else
      fprintf( 2, '\n\t Error: tmp kernel file %s does not exist\n', ...
        conf.tmpKernelPath{ jobID } );
      allExist = false;
    end
  end
  % get training and testing kernel matrix
  % selTrain = ( imdb.ttSplit == 1 );
  % selTest =  ( imdb.ttSplit == 0 );
  % numTrain = sum( selTrain );
  % numTest = sum( selTest );
  % kernelTrain = [ ( 1 : numTrain )', ...
  %   kernel( selTrain, selTrain ) ];
  % kernelTest = [ ( 1 : numTest )', ...
  %   kernel( selTest, selTrain ) ];

  % save all kernel
  if( allExist )
    save( conf.kernelPath, 'kernel', '-v7.3'  );
  else
    fprintf( '\n\t Error: kernel incomplete!\n' );
  end
end

fprintf( '\n ...Done Aggregate Precomputed Kernel time: %.2f (s)',  toc );
