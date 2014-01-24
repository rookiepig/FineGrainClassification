%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: part_kernel.m
% Desc: split each part kernel
% Author: Zhang Kang
% Date: 2014/01/13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function part_kernel( jobID, JOB_NUM, partID )

% Step3: aggregate features
tic;
fprintf( '\n convert desct to part kernel...\n' );
fprintf( 'Part ID: %d\n', partID );

% initial all configuration
initConf;
% temporary encoding files
conf.cacheDir = [ 'cache/' conf.dataset ];            % cache dir for temp files
conf.jobNum = JOB_NUM;
for ii = 1 : conf.jobNum
  tempFn = sprintf( '-tmpDescrs%03d.mat', ii );
  conf.tmpDescrsPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end

conf.tmpKernelPath = cell( 1, conf.jobNum );
for ii = 1 : conf.jobNum
  tempFn = sprintf( '-tmpKernel%03dPart%d.mat', ii, partID );
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
if( exist( conf.tmpDescrsPath{ jobID }, 'file' ) ) % descrs file exist
  
  if( ~exist( conf.tmpKernelPath{ jobID }, 'file' ) )
    fprintf( '\n\t Kernel job: %03d (%.2f %%) ... \n', ...
      jobID, 100 * jobID / conf.jobNum  );
    % load current job des
    load( conf.tmpDescrsPath{ jobID } );
    rowDes = cat( 2, jobDes{ : } );
    ttFV = size( rowDes, 1 );
    fprintf( 'total FV len: %d\n', ttFV );
    ptFV = floor( ttFV / 5 );
    fprintf( 'part FV len: %d\n', ptFV );
    ptSt = ( partID - 1 ) * ptFV + 1;
    ptEd = partID * ptFV;
    rowDes = rowDes( ptSt : ptEd, : );

    ttImgNum = numel( imdb.imgName );
    jobSz = floor( ttImgNum / conf.jobNum );
    rowSt = ( jobID - 1 ) * jobSz + 1;
    if( jobID == conf.jobNum )
      rowEd = ttImgNum;
    else
      rowEd = jobID * jobSz;
    end
    jobKernel = zeros( rowEd - rowSt + 1, ttImgNum );
    
    % compute precomputed kernel
    for colID = 1 : conf.jobNum
      if( exist( conf.tmpDescrsPath{ colID }, 'file' ) )
        fprintf( '\n\t\t load col descrs: %s (%.2f %%)\n', ...
          conf.tmpDescrsPath{ colID }, 100 * colID / conf.jobNum  );
        % load current job des
        load( conf.tmpDescrsPath{ colID } );
        
        colSt = ( colID - 1 ) * jobSz + 1;
        if( colID == conf.jobNum )
          colEd = ttImgNum;
        else
          colEd = colID * jobSz;
        end
        colDes = cat( 2, jobDes{ : } );
        clear jobDes; % save memory
        % block matrix multiply
        colDes = colDes( ptSt : ptEd, : );
        jobKernel( :, colSt : colEd ) = ...
          rowDes' * colDes;
      else
        fprintf( 2, '\n\t\t Error: col file %s does not exist\n', ...
          conf.tmpDescrsPath{ colID } );
      end
    end
    
    % save job kernel
    save( conf.tmpKernelPath{ jobID }, 'jobKernel' );
  end
else
  fprintf( 2, '\n\t Error: row file %s does not exist\n', ...
    conf.tmpDescrsPath{ jobID } );
end


fprintf( '\n ...Done Kernel job: %03d, time: %.2f (s)', jobID, toc );
