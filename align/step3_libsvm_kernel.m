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
    ttImgNum = numel( imdb.imgName );
    kernelAll = zeros( ttImgNum, ttImgNum );
    jobSz = floor( ttImgNum / conf.jobNum );
    for rowID = 1 : conf.jobNum
        if( exist( conf.descrsPath{ rowID }, 'file' ) )
            fprintf( '\n\t row descrs: %s (%.2f %%)\n', ... 
                conf.descrsPath{ rowID }, 100 * rowID / conf.jobNum  );
            % load current job des
            load( conf.descrsPath{ rowID } );
            
            rowSt = ( rowID - 1 ) * jobSz + 1;
            if( rowID == conf.jobNum )
                rowEd = ttImgNum;
            else
                rowEd = rowID * jobSz;
            end
            rowDes = cat( 2, jobDes{ : } );
            
            for colID = 1 : conf.jobNum
                if( exist( conf.descrsPath{ colID }, 'file' ) )
                    fprintf( '\n\t\t col descrs: %s (%.2f %%)\n', ... 
                        conf.descrsPath{ colID }, 100 * colID / conf.jobNum  );
                    % load current job des
                    load( conf.descrsPath{ colID } );

                    colSt = ( colID - 1 ) * jobSz + 1;
                    if( colID == conf.jobNum )
                        colEd = ttImgNum;
                    else
                        colEd = colID * jobSz;
                    end
                    colDes = cat( 2, jobDes{ : } );
                    % block matrix multiply
                    kernelAll( rowSt : rowEd, colSt : colEd ) = ...
                        rowDes' * colDes;
                else
                    fprintf( 2, 'Error: descrs file %s does not exist\n', ... 
                        conf.descrsPath{ colID } );
                end
            end
        else
            fprintf( 2, 'Error: descrs file %s does not exist\n', ... 
                conf.descrsPath{ rowID } );
        end
    end
    
    % compute kernel matrix
    selTrain = ( imdb.ttSplit == 1 );
    selTest =  ( imdb.ttSplit == 0 );
    numTrain = sum( selTrain );
    numTest = sum( selTest );
    kernelTrain = [ ( 1 : numTrain )', ...
        kernelAll( selTrain, selTrain ) ];
    kernelTest = [ ( 1 : numTest )', ...
        kernelAll( selTest, selTrain ) ];
    save( conf.featPath, 'kernelTrain', 'kernelTest'  );

end

fprintf( '\n ...Done Step3: Aggregate Features and Compute Kernel time: %.2f (s)',  toc );
