%% TrainEncoder: train image encoder
% -----------------------------------------------------------------
function [encoder] = TrainEncoder( imgList, varargin )
% -----------------------------------------------------------------
fprintf( 1, '\n Train Encoder ... \n' );
% declare global variables
global conf imdb;

if ( nargin == 2 )
	% 1 paramter --> bounding box list
	boxList = varargin{ 1 };
elseif( nargin == 3 )
	% 2 paramters --> bounding box + mask list
	boxList = varargin{ 1 };
	maskList = varargin{ 2 };
end

% default options for encoder
opts.type = 'fv' ;
opts.numWords = [];
opts.seed = 1;
opts.numPcaDimensions = +inf ;
opts.whitening = false ;
opts.whiteningRegul = 0;
opts.numSamplesPerWord = [];
opts.renormalize =  true;
opts.layouts = {'1x1'} ;
opts.subdivisions = zeros(4,0) ;
opts.readImgFunc = @ReadImg;
opts.getFeatFunc = @GetFeat; 
% set paramters use global configuration
opts = vl_argparse( opts, conf.encoderParam ) ;

% spatial pyramid layout --> subdivision
for i = 1:numel(opts.layouts)
  t = sscanf(opts.layouts{i},'%dx%d') ;
  m = t(1) ;
  n = t(2) ;
  [x,y] = meshgrid(...
    linspace(0,1,n+1), ...
    linspace(0,1,m+1)) ;
  x1 = x(1:end-1,1:end-1) ;
  y1 = y(1:end-1,1:end-1) ;
  x2 = x(2:end,2:end) ;
  y2 = y(2:end,2:end) ;
  opts.subdivisions = cat(2, opts.subdivisions, ...
    [x1(:)' ;
     y1(:)' ;
     x2(:)' ;
     y2(:)'] ) ;
end

% default visual word
if isempty( opts.numWords )
    switch opts.type
      case {'bovw'}
        opts.numWords = 1024 ;
      case {'fv'} % fv setting follows [Gavves, ICCV13]
        opts.numWords = 256 ;
        opts.numPcaDimensions = 64;
      case {'vlad'}
        opts.numWords = 64 ;
        opts.numPcaDimensions = 100 ;
        opts.whitening = true ;
        opts.whiteninRegul = 0.01 ;
      otherwise
        assert(false) ;
    end
end

if isempty(opts.numSamplesPerWord)
    switch opts.type
      case {'bovw'}
        opts.numSamplesPerWord = 200;
      case {'vlad','fv'}
        opts.numSamplesPerWord = 1000;
      otherwise
        assert(false) ;
    end
    if conf.lite
      opts.numSamplesPerWord = 20;
    end
end

% show final options and save to encoder
disp(opts) ;

encoder.type = opts.type ;
encoder.subdivisions = opts.subdivisions ;
encoder.numWords = opts.numWords ;
encoder.renormalize = opts.renormalize ;
encoder.readImgFunc = opts.readImgFunc;
encoder.getFeatFunc = opts.getFeatFunc;

%% Step 0: obtain sample image descriptors
%% if mask is available --> mask out descriptors

numImages = numel( imgList ) ;
numDescrsPerImage = ceil(opts.numWords * opts.numSamplesPerWord / numImages) ;
descrs = cell( 1, numImages );
for ii = 1 : numImages
  fprintf('\n\t%s: reading: %s (%.2f %%)', mfilename,  imgList{ ii }, ...
      100 * ii / numImages ) ;
  if( conf.useBoundingBox )
  	img = encoder.readImgFunc(  imgList { ii }, boxList( ii, : ) ) ;
  else
  	img = encoder.readImgFunc(  imgList { ii } ) ;
  end
  if( conf.useSegMask )
  	% treat each part equally
	mask = encoder.readImgFunc( maskList{ ii }, boxList( ii, : ) );
  	features = encoder.getFeatFunc( img, mask ) ;
  else
  	features = encoder.getFeatFunc( img );
  end
  % sampling
  randn( 'state', 0 );
  rand( 'state', 0 );
  sel = vl_colsubset( 1 : size( features.descr, 2 ), ...
  	single( numDescrsPerImage ) );
  descrs{ ii } = features.descr( :, sel );
end
descrs = cat( 2, descrs{ : } );

%% Step 1 (optional): learn PCA projection
dimension = size( descrs, 1 ) ;
% each channel has its own PCA
dimPerCh = conf.featDimPerChannel;
siftNum = floor( dimension / dimPerCh );
fprintf('\n %s: PCA num %d\n', mfilename, siftNum ) ;
encoder.pcaNum = siftNum;

for sn = 1 : siftNum  
  if opts.numPcaDimensions < inf || opts.whitening
    fprintf('\n\t%s: learning PCA rotation/projection %d\n', ...
      mfilename, sn ) ;
    desSt = ( sn - 1 ) * dimPerCh + 1;
    desEd = sn * dimPerCh;
    curDescrs = descrs( desSt : desEd, : );
    encoder.projectionCenter{ sn } = mean( curDescrs, 2 ) ;
    x = bsxfun( @minus, curDescrs, encoder.projectionCenter{ sn } ) ;
    X = x*x' / size(x,2) ;
    [V,D] = eig(X) ;
    d = diag(D) ;
    [d,perm] = sort(d,'descend') ;
    d = d + opts.whiteningRegul * max(d) ;
    m = min(opts.numPcaDimensions, size(curDescrs,1)) ;
    V = V(:,perm) ;
    if opts.whitening
      encoder.projection{ sn } = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
    else
      encoder.projection{ sn } = V(:,1:m)' ;
    end
    clear X V D d ;
  else
    encoder.projection{ sn } = 1 ;
    encoder.projectionCenter{ sn } = 0 ;
  end
end

% project descrs using PCA
for pn = 1 : encoder.pcaNum
  desSt = ( pn - 1 ) * dimPerCh + 1;
  desEd = pn * dimPerCh;
  curDescrs = descrs( desSt : desEd, : );
  curDescrs = encoder.projection{ pn } * ...
    bsxfun(@minus, curDescrs, encoder.projectionCenter{ pn } ) ;
  if encoder.renormalize
    curDescrs = bsxfun(@times, curDescrs, 1./max(1e-12, sqrt(sum(curDescrs.^2)))) ;
  end
  tmpDescrs{ pn } = curDescrs;
end
descrs = cat( 1, tmpDescrs{ : } );

%% Step 2 (optional): geometrically augment the features
% add image coordinates as features

% descrs = extendDescriptorsWithGeometry(opts.geometricExtension, frames, descrs) ;

%% Step 3: learn a VQ or GMM vocabulary
dimension = size( descrs, 1 ) ;
numDescriptors = size( descrs, 2 ) ;
fprintf( '\n\t descrs dim: %d', dimension );
fprintf( '\n\t descrs num: %d', numDescriptors );

switch encoder.type
  case {'bovw', 'vlad'}
    vl_twister('state', opts.seed) ;
    encoder.words = vl_kmeans(descrs, opts.numWords, 'verbose', 'algorithm', 'elkan') ;
    encoder.kdtree = vl_kdtreebuild(encoder.words, 'numTrees', 2) ;

  case {'fv'} ;
    vl_twister('state', opts.seed) ;
    if 1
        
      v = var(descrs')' ;
      if( conf.lite )
          % for debug output v and descrs
          save( fullfile( conf.outDir, ... 
            [ conf.prefix, 'gmmDescrs.mat'] ), 'descrs' );
          save( fullfile( conf.outDir, ...
            [ conf.prefix,'gmmVar.mat' ] ), 'v' );
      end
      % vl_feat gmm
      [encoder.means, encoder.covariances, encoder.priors] = ...
          vl_gmm(descrs, opts.numWords, 'verbose', ...
                 'Initialization', 'kmeans', ...
                 'CovarianceBound', double(max(v)*0.0001), ...
                 'NumRepetitions', 1) ;
          
      % try matlab gmm
      % gmOpt = statset('Display','iter');
      % gmObj = gmdistribution.fit( descrs', opts.numWords, ...
      %     'CovType', 'diagonal', ...
      %     'Regularize', double( max( v ) ) * 0.0001, ...
      %     'Options', gmOpt );
      % encoder.means = gmObj.mu';
      % encoder.covariances = reshape( gmObj.Sigma, dimension, opts.numWords );
      % encoder.priors = single( gmObj.PComponents' );    
    else
      addpath lib/yael/matlab
      [a,b,c] = ...
          yael_gmm(descrs, opts.numWords, 'verbose', 2) ;
      encoder.priors = single(a) ;
      encoder.means = single(b) ;
      encoder.covariances = single(c) ;
    end
end

fprintf( '\n ... Done\n' );



