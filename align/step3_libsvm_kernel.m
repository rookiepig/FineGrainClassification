%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step3_libsvm_kernel.m
% Desc: compute different kernel maps
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step3_libsvm_kernel( jobID, JOB_NUM )

% Step3: aggregate features
tic;
fprintf( '\n Step3: Precompute Kernel...\n' );

% initial all configuration
initConf;
% temporary encoding files
conf.cacheDir = 'cache';                  % cache dir for temp files
conf.jobNum = JOB_NUM;                         
for ii = 1 : conf.jobNum
    tempFn = sprintf( '-tmpDescrs%03d.mat', ii );
    conf.tmpDescrsPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end

conf.tmpKernelPath = cell( 1, conf.jobNum );
for ii = 1 : conf.jobNum
    tempFn = sprintf( '-tmpKernel%03d.mat', ii );
    conf.tmpKernelPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end


% setup dataset
setupCUB11;

% load econded features
if( ~exist( conf.kernelPath, 'file' )  ) % kernel file not exist

    if( exist( conf.tmpDescrsPath{ jobID }, 'file' ) ) % descrs file exist

        if( ~exist( conf.tmpKernelPath{ jobID }, 'file' ) )
            fprintf( '\n\t Kernel job: %03d (%.2f %%) ... \n', ... 
                jobID, 100 * jobID / conf.jobNum  );
            % load current job des
            load( conf.tmpDescrsPath{ jobID } );
            rowDes = cat( 2, jobDes{ : } );

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
                    % block matrix multiply
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

end

fprintf( '\n ...Done Kernel job: %03d, time: %.2f (s)', jobID, toc );
