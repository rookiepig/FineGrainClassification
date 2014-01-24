%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: part_aggre.m
% Desc: aggregate kernel maps
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function part_aggre( JOB_NUM, partID )

% Step3: aggregate features
tic;
fprintf( '\n aggregate part kernel\n' );
fprintf( 'Part ID: %d\n', partID );


% initial all configuration
initConf;
% temporary encoding files
conf.cacheDir = [ 'cache/' conf.dataset ];            % cache dir for temp files
conf.jobNum = JOB_NUM;
conf.tmpKernelPath = cell( 1, conf.jobNum );
for ii = 1 : conf.jobNum
  tempFn = sprintf( '-tmpKernel%03dPart%d.mat', ii, partID );
  conf.tmpKernelPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end
% part kernel file
partKernelPath = sprintf( '-kernel-part%d.mat', partID );
partKernelPath = fullfile( conf.outDir, [conf.prefix partKernelPath] );

% setup dataset
switch conf.dataset
  case {'CUB11'}
    setupCUB11;
  case {'STDog'}
    setupSTDog;
end

% load econded features
ttImgNum = numel( imdb.imgName );
kernel = zeros( ttImgNum, ttImgNum );
jobSz = floor( ttImgNum / conf.jobNum );

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
  end
end
% save all kernel
save( partKernelPath, 'kernel', '-v7.3'  );


fprintf( '\n ...Done Aggregate Precomputed Kernel time: %.2f (s)',  toc );
