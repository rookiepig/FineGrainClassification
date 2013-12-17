%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step3_aggrFeat.m
% Desc: aggregate different encoding results
% Author: Zhang Kang
% Date: 2013/12/06
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step3_aggrFeat( JOB_NUM )

% Step3: aggregate features
tic;
fprintf( '\n Step3: Aggregate Features ...\n' );


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
    fprintf( '\n\t aggregate feature exist: %s', conf.featPath );
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
            clear jobDes;
            
        else
            fprintf( 2, 'Error: descrs file %s does not exist\n', ... 
                conf.descrsPath{ jobID } );
            exit;
        end
    end
    % use sparse matrix to save memory
    descrs =  cat( 2, descrs{ : } );
    % fprintf( '\n\t descrs sparseness: %.2f %%', ...
    %     100 * length( find( abs( descrs ) > 1e-6 ) ) / numel( descrs ) );
    save( conf.featPath, 'descrs', '-v7.3'  );

    % delete temp descrs files
    for jobID = 1 : conf.jobNum
        if( exist( conf.descrsPath{ jobID }, 'file' ) )
            delete( conf.descrsPath{ jobID } );
        end
    end
end

fprintf( '\n ...Done Step3: Aggregate Features time: %.2f (s)',  toc );
