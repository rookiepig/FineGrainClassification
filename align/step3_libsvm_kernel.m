%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step3_libsvm_kernel.m
% Desc: aggregate different encoding results and compute kernel map
% Author: Zhang Kang
% Date: 2013/12/15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step3_libsvm_kernel( JOB_NUM )

% Step3: aggregate features
tic;
fprintf( '\n Step3: Aggregate Features and Compute Kernel...\n' );


% initial all configuration
initConf;
% temporary encoding files
conf.jobNum = JOB_NUM;                         % parallel jobs for encoding
conf.descrsPath = cell( 1, conf.jobNum );
for ii = 1 : conf.jobNum
    tempFn = sprintf( '-descrs%03d.mat', ii );
    conf.descrsPath{ ii } = fullfile( conf.outDir, [conf.prefix tempFn] );
end

% setup dataset
setupCUB11;

% load econded features
if( exist( conf.featPath, 'file' ) )
    fprintf( '\n\t precompute kernel file exist: %s', conf.featPath );
else
    descrs = cell( 1, numel( imdb.imgName ) );
    ttImgNum = numel( imdb.imgName );
    jobSz = floor( ttImgNum / conf.jobNum );
    for jobID = 1 : conf.jobNum
        if( exist( conf.descrsPath{ jobID }, 'file' ) )
            fprintf( '\n\t Load descrs: %s (%.2f %%)\n', ... 
                conf.descrsPath{ jobID }, 100 * jobID / conf.jobNum  );
            % load current job des
            load( conf.descrsPath{ jobID } );
            
            jobSt = ( jobID - 1 ) * jobSz + 1;
            if( jobID == conf.jobNum )
                jobEd = ttImgNum;
            else
                jobEd = jobID * jobSz;
            end
            
            descrs( 1, jobSt : jobEd ) = jobDes;
            % remove temp descrs files
            clear jobDes;
        else
            fprintf( 2, 'Error: descrs file %s does not exist\n', ... 
                conf.descrsPath{ jobID } );
        end
    end
    % use sparse matrxi to save memory
    descrs =  cat( 2, descrs{ : } );
    % fprintf( '\n\t descrs sparseness: %.2f %%', ...
    %     100 * length( find( abs( descrs ) > 1e-6 ) ) / numel( descrs ) );
    
    size( descrs )
    
    % compute kernel matrix
    selTrain = ( imdb.ttSplit == 1 );
    selTest =  ( imdb.ttSplit == 0 );
    numTrain = sum( selTrain );
    numTest = sum( selTest );
    kernelTrain = [ ( 1 : numTrain )', ...
        descrs( : , selTrain )' * descrs( : , selTrain ) ];
    kernelTest = [ ( 1 : numTest )', ...
        descrs( : , selTest )' * descrs( : , selTrain ) ];
    save( conf.featPath, 'kernelTrain', 'kernelTest'  );
    
    % remvoe temp descrs file
    for jobID = 1 : conf.jobNum
        if( exist( conf.descrsPath{ jobID }, 'file' ) )
            delete( conf.descrsPath{ jobID } );
        end
    end
end

fprintf( '\n ...Done Step3: Aggregate Features and Compute Kernel time: %.2f (s)',  toc );
