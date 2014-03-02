%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step2_encoding.m
% Desc: parallel running encoding
% Author: Zhang Kang
% Date: 2013/12/08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function step2_encoding( jobID, JOB_NUM )

% Step2: encoding
tic;
fprintf( '\n Step2: Encoding images ...\n' );

% init  configuration
initConf;
% setup dataset
switch conf.dataset
  case {'CUB11'}
    setupCUB11;
  case {'STDog'}
    setupSTDog;
end

% temporary encoding files
conf.cacheDir = [ 'cache/' conf.dataset ];            % cache dir for temp files
conf.jobNum = JOB_NUM;                         % parallel jobs for encoding
conf.tmpDescrsPath = cell( 1, conf.jobNum );
for ii = 1 : conf.jobNum
  tempFn = sprintf( '-tmpDescrs%03d.mat', ii );
  conf.tmpDescrsPath{ ii } = fullfile( conf.cacheDir, [conf.prefix tempFn] );
end

% split image according to jobID and jobNum
ttImgNum = numel( imdb.imgName );
jobSz = floor( ttImgNum / conf.jobNum );
jobSt = ( jobID - 1 ) * jobSz + 1;
if( jobID == conf.jobNum )
  jobEd = ttImgNum;
else
  jobEd = jobID * jobSz;
end

if exist( conf.encoderPath, 'file' )
  % load existing encoder
  fprintf( '\n Get encoder from: %s\n', conf.encoderPath );
  load( conf.encoderPath ) ;
  
  fprintf( '\n\t Encoding job: %03d (%.2f %%)...\n', jobID, ...
    100 * jobID / conf.jobNum );
  
  % encoding current job images
  if( ~exist( conf.tmpDescrsPath{ jobID }, 'file' ) )
    
    jobDes = cell( 1, ( jobEd - jobSt ) + 1 );
    for ii = jobSt : jobEd
      fprintf( '\n\t encoding image %s (%.2f %%)', imdb.imgName{ ii }, ...
        100 * ii / numel( imdb.imgName ) );
      if( conf.useSegMask )
        jobDes{ ii - jobSt + 1 } = EncodeImg( encoder, ...
          fullfile( imdb.imgDir, imdb.imgName{ ii } ), ...
          imdb.bdBox( ii, : ), ...
          fullfile( imdb.imgDir, imdb.maskName{ ii } ) ) ;
      elseif( conf.useBoundingBox )
        jobDes{ ii - jobSt + 1 } = EncodeImg( encoder, ...
          fullfile( imdb.imgDir, imdb.imgName{ ii } ), ...
          imdb.bdBox( ii, : ) ) ;
      end
    end
    
    % save job des
    save( conf.tmpDescrsPath{ jobID }, 'jobDes' );
  end
  fprintf( '\n ... Done\n' );
  
else
  % must train encoder first
  fprintf( 2, '\n Error: encoder file: %s does not exist\n', conf.encoderPath );
end

fprintf( '\n ...Done Encoding job: %03d, time: %.2f (s)', jobID, toc );